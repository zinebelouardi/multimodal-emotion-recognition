"""
data_loader.py
--------------
Chargement et préparation des données CREMA-D (audio + vidéo).
"""

import os
import pandas as pd


# Mapping des codes émotion vers des labels lisibles
EMOTION_MAP = {
    'ANG': 'angry',
    'DIS': 'disgust',
    'FEA': 'fear',
    'HAP': 'happy',
    'NEU': 'neutral',
    'SAD': 'sad'
}


def load_video_metadata(video_root: str) -> pd.DataFrame:
    """
    Parcourt le dossier vidéo et retourne un DataFrame avec les métadonnées.

    Args:
        video_root: Chemin vers le dossier contenant les fichiers .mp4

    Returns:
        DataFrame avec colonnes: video_path, actor_id, statement, emotion, intensity, gender
    """
    data = []

    for root, dirs, files in os.walk(video_root):
        for file in files:
            if file.endswith(".mp4"):
                file_path = os.path.join(root, file)
                parts = file.replace(".mp4", "").split("_")

                actor_id  = parts[0]
                statement = parts[1]
                emotion   = parts[2]
                intensity = parts[3]
                gender_code = parts[4]
                gender    = "male" if gender_code == "01" else "female"

                data.append([file_path, actor_id, statement, emotion, intensity, gender])

    df = pd.DataFrame(data, columns=[
        "video_path", "actor_id", "statement", "emotion", "intensity", "gender"
    ])
    return df


def load_multimodal_dataframe(video_root: str, audio_root: str) -> pd.DataFrame:
    """
    Construit un DataFrame multimodal en associant chaque fichier audio
    à son fichier vidéo correspondant via une clé commune.

    Args:
        video_root: Chemin vers le dossier contenant les fichiers .mp4
        audio_root: Chemin vers le dossier contenant les fichiers .wav

    Returns:
        DataFrame avec colonnes: audio_key, audio_path, emotion, video_path
    """
    # --- Audio ---
    audio_data = []
    for file in os.listdir(audio_root):
        if file.endswith('.wav'):
            file_key = file.split('.')[0]           # ex: 1001_DFA_ANG_XX
            emotion_code = file_key.split('_')[2]   # ex: ANG
            audio_data.append({
                'audio_key':  file_key,
                'audio_path': os.path.join(audio_root, file),
                'emotion':    EMOTION_MAP.get(emotion_code, 'unknown')
            })

    df_audio = pd.DataFrame(audio_data)

    # --- Vidéo ---
    video_data = []
    for root, dirs, files in os.walk(video_root):
        for file in files:
            if file.endswith('.mp4'):
                # Supprimer le suffixe numérique final: 1001_DFA_ANG_XX_01 -> 1001_DFA_ANG_XX
                video_key = "_".join(file.split('_')[:-1])
                video_data.append({
                    'audio_key':  video_key,
                    'video_path': os.path.join(root, file)
                })

    df_video = pd.DataFrame(video_data)

    # --- Fusion ---
    df_multimodal = pd.merge(df_audio, df_video, on='audio_key', how='inner')

    print(f"Total paires multimodales trouvées : {len(df_multimodal)}")
    return df_multimodal

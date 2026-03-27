"""
dataset.py
----------
Classe PyTorch Dataset pour le chargement des paires (vidéo, audio, label).
"""

import cv2
import torch
import numpy as np
import torchaudio
import torchvision.transforms as T
from torch.utils.data import Dataset


class CremaDataset(Dataset):
    """
    Dataset PyTorch pour CREMA-D (audio + vidéo multimodal).

    Args:
        df:             DataFrame avec colonnes 'video_path', 'audio_path', 'emotion'
        seq_len:        Nombre de frames à extraire par vidéo
        audio_duration: Durée audio en secondes
    """

    def __init__(self, df, seq_len: int = 8, audio_duration: int = 5):
        self.df             = df.reset_index(drop=True)
        self.seq_len        = seq_len
        self.audio_duration = audio_duration
        self.target_sr      = 16000
        self.audio_samples  = self.target_sr * audio_duration

        self.emotions  = sorted(df['emotion'].unique())
        self.label_map = {label: i for i, label in enumerate(self.emotions)}

        # Transformations vidéo
        self.video_transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

        # Spectrogramme Mel
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,
            n_mels=128,
            n_fft=400,
            hop_length=160,
            power=2.0
        )

    def __len__(self) -> int:
        return len(self.df)

    def _get_video_frames(self, path: str) -> torch.Tensor:
        """Extrait seq_len frames uniformément réparties dans la vidéo."""
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames == 0:
            cap.release()
            return torch.zeros(self.seq_len, 3, 224, 224)

        indices = set(np.linspace(0, total_frames - 1, self.seq_len, dtype=int))
        frames  = []

        for i in range(total_frames):
            ret, frame = cap.read()
            if i in indices and ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.video_transform(frame))

        cap.release()

        # Padding si nécessaire
        while len(frames) < self.seq_len:
            frames.append(torch.zeros(3, 224, 224))

        return torch.stack(frames[:self.seq_len])  # [seq_len, 3, 224, 224]

    def _get_audio_spec(self, path: str) -> torch.Tensor:
        """Charge l'audio et retourne un spectrogramme Mel [1, 128, 128]."""
        try:
            waveform, sr = torchaudio.load(path)

            # Mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample
            if sr != self.target_sr:
                waveform = torchaudio.transforms.Resample(sr, self.target_sr)(waveform)

            # Ajuster la longueur
            if waveform.shape[1] < self.audio_samples:
                pad = self.audio_samples - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad))
            else:
                waveform = waveform[:, :self.audio_samples]

            # Spectrogramme Mel + dB
            mel = self.mel_spec(waveform)
            mel = torchaudio.transforms.AmplitudeToDB()(mel)

            # Redimensionner à [1, 128, 128] si nécessaire
            if mel.shape[2] != 128:
                mel = torch.nn.functional.interpolate(
                    mel.unsqueeze(0), size=(128, 128)
                ).squeeze(0)

            return mel

        except Exception as e:
            print(f"[WARNING] Erreur chargement audio {path}: {e}")
            return torch.zeros(1, 128, 128)

    def __getitem__(self, idx: int):
        row   = self.df.iloc[idx]
        video = self._get_video_frames(row['video_path'])
        audio = self._get_audio_spec(row['audio_path'])
        label = self.label_map[row['emotion']]
        return video, audio, label

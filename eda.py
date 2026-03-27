"""
eda.py
------
Analyse exploratoire des données (EDA) du dataset CREMA-D.
"""

import matplotlib.pyplot as plt
import pandas as pd


COLUMNS_TO_ANALYZE = ["statement", "emotion", "intensity", "gender"]


def run_eda(df: pd.DataFrame) -> None:
    """
    Affiche les statistiques et graphiques pour chaque colonne catégorielle.

    Args:
        df: DataFrame contenant au moins les colonnes définies dans COLUMNS_TO_ANALYZE
    """
    for col in COLUMNS_TO_ANALYZE:
        if col not in df.columns:
            print(f"[WARNING] Colonne '{col}' absente du DataFrame — ignorée.")
            continue

        print(f"\n{'='*16} {col.upper()} {'='*16}")
        print("Valeurs uniques :")
        print(df[col].unique())
        print("\nCompte par valeur :")
        print(df[col].value_counts())

        # Bar chart
        plt.figure()
        df[col].value_counts().plot(kind="bar")
        plt.title(f"Distribution de '{col}'")
        plt.xlabel(col)
        plt.ylabel("Nombre d'échantillons")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        # Pie chart
        plt.figure()
        df[col].value_counts().plot(kind="pie", autopct="%1.1f%%")
        plt.title(f"Répartition de '{col}'")
        plt.ylabel("")
        plt.tight_layout()
        plt.show()

"""
src — package principal du projet Multimodal Emotion Detection.
"""

from src.data_loader import load_multimodal_dataframe, load_video_metadata
from src.dataset import CremaDataset
from src.model import MultimodalLSTM
from src.train import train_one_epoch, evaluate
from src.eda import run_eda

__all__ = [
    "load_multimodal_dataframe",
    "load_video_metadata",
    "CremaDataset",
    "MultimodalLSTM",
    "train_one_epoch",
    "evaluate",
    "run_eda",
]

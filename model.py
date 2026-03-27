"""
model.py
--------
Architecture MultimodalLSTM :
  - Backbone vidéo  : EfficientNet-B0 (timm)
  - Backbone audio  : ResNet18      (timm)
  - Fusion temporelle : LSTM bidirectionnel
  - Classificateur  : MLP 2 couches
"""

import torch
import torch.nn as nn
import timm


class MultimodalLSTM(nn.Module):
    """
    Modèle multimodal pour la reconnaissance d'émotions.

    Args:
        num_classes:     Nombre d'émotions à prédire
        video_feat_dim:  Dimension des features vidéo après projection
        audio_feat_dim:  Dimension des features audio après projection
    """

    def __init__(
        self,
        num_classes:     int = 6,
        video_feat_dim:  int = 512,
        audio_feat_dim:  int = 512,
    ):
        super().__init__()

        # ── Backbones ────────────────────────────────────────────────────────
        self.video_backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,   # supprime la tête de classification
            in_chans=3
        )

        self.audio_backbone = timm.create_model(
            'resnet18',
            pretrained=True,
            num_classes=0,
            in_chans=1       # spectrogramme mono
        )

        # ── Projections ───────────────────────────────────────────────────────
        self.video_proj = nn.Linear(1280, video_feat_dim)   # EfficientNet-B0 → 1280
        self.audio_proj = nn.Linear(512,  audio_feat_dim)   # ResNet18        → 512

        # ── Fusion temporelle ─────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=video_feat_dim + audio_feat_dim,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=False
        )

        # ── Classificateur ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, video: torch.Tensor, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [batch, seq_len, 3, 224, 224]
            audio: [batch, 1, 128, 128]

        Returns:
            logits: [batch, num_classes]
        """
        batch_size, seq_len = video.shape[:2]

        # --- Vidéo (frame par frame) ---
        video_flat     = video.view(-1, 3, 224, 224)                      # [B*T, 3, 224, 224]
        video_features = self.video_backbone(video_flat)                   # [B*T, 1280]
        video_features = self.video_proj(video_features)                   # [B*T, D_v]
        video_features = video_features.view(batch_size, seq_len, -1)     # [B, T, D_v]

        # --- Audio (global) ---
        audio_features = self.audio_backbone(audio)                        # [B, 512]
        audio_features = self.audio_proj(audio_features)                   # [B, D_a]
        audio_features = audio_features.unsqueeze(1).repeat(1, seq_len, 1)# [B, T, D_a]

        # --- Fusion + LSTM ---
        combined  = torch.cat([video_features, audio_features], dim=2)    # [B, T, D_v+D_a]
        lstm_out, _ = self.lstm(combined)                                  # [B, T, 256]

        # Dernier timestep
        final = lstm_out[:, -1, :]                                         # [B, 256]

        return self.classifier(final)                                      # [B, num_classes]

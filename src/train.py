"""
train.py
--------
Entraînement du modèle MultimodalLSTM sur CREMA-D.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from src.data_loader import load_multimodal_dataframe
from src.dataset import CremaDataset
from src.model import MultimodalLSTM


# ─── Chemins (à adapter) ────────────────────────────────────────────────────
VIDEO_ROOT  = '/kaggle/input/datasets/alenken/multimodal-emotion-recognition-ravdess/crema-d'
AUDIO_ROOT  = '/kaggle/input/datasets/ejlok1/cremad/AudioWAV'

# ─── Hyperparamètres ─────────────────────────────────────────────────────────
BATCH_SIZE  = 8
NUM_EPOCHS  = 10
LR          = 1e-4
WEIGHT_DECAY = 0.01
SEQ_LEN     = 8
TEST_SIZE   = 0.2
SEED        = 42


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct = 0.0, 0

    pbar = tqdm(loader, desc="  Train", leave=False)
    for video, audio, labels in pbar:
        video, audio, labels = video.to(device), audio.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(video, audio)
        loss    = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        correct    += (outputs.argmax(1) == labels).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader), correct / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct = 0.0, 0
    all_preds, all_targets = [], []

    for video, audio, labels in tqdm(loader, desc="  Val  ", leave=False):
        video, audio, labels = video.to(device), audio.to(device), labels.to(device)
        outputs = model(video, audio)
        loss    = criterion(outputs, labels)

        total_loss  += loss.item()
        correct     += (outputs.argmax(1) == labels).sum().item()
        all_preds   .extend(outputs.argmax(1).cpu().numpy())
        all_targets .extend(labels.cpu().numpy())

    return total_loss / len(loader), correct / len(loader.dataset), all_preds, all_targets


def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'],   label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'],   label='Val Acc')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('outputs/training_curves.png', dpi=150)
    plt.show()


def plot_confusion_matrix(all_targets, all_preds, emotion_names):
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay.from_predictions(
        all_targets, all_preds,
        display_labels=emotion_names,
        ax=ax,
        cmap='Blues'
    )
    plt.title('Matrice de confusion')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png', dpi=150)
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # ── Données ──────────────────────────────────────────────────────────────
    df = load_multimodal_dataframe(VIDEO_ROOT, AUDIO_ROOT)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df['emotion'],
        random_state=SEED
    )
    print(f"Train : {len(train_df)} | Test : {len(test_df)}")
    print(f"Émotions : {sorted(train_df['emotion'].unique())}")

    train_ds = CremaDataset(train_df, seq_len=SEQ_LEN)
    test_ds  = CremaDataset(test_df,  seq_len=SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # ── Modèle ───────────────────────────────────────────────────────────────
    num_classes = len(train_df['emotion'].unique())
    model       = MultimodalLSTM(num_classes=num_classes).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres totaux : {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # ── Test forward pass ────────────────────────────────────────────────────
    model.eval()
    try:
        video, audio, _ = next(iter(train_loader))
        video, audio = video.to(device), audio.to(device)
        with torch.no_grad():
            out = model(video, audio)
        print(f"✓ Forward pass OK — output shape : {out.shape}")
    except Exception as e:
        print(f"✗ Forward pass échoué : {e}")
        return

    # ── Entraînement ─────────────────────────────────────────────────────────
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    print("\n" + "=" * 50)
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")

        t_loss, t_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        v_loss, v_acc, all_preds, all_targets = evaluate(model, test_loader, criterion, device)

        history['train_loss'].append(t_loss)
        history['train_acc'] .append(t_acc)
        history['val_loss']  .append(v_loss)
        history['val_acc']   .append(v_acc)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), 'outputs/best_model.pth')
            print(f"  ✓ Nouveau meilleur modèle sauvegardé (val_acc={v_acc:.4f})")

        print(f"  Train Loss={t_loss:.4f} Acc={t_acc:.4f} | Val Loss={v_loss:.4f} Acc={v_acc:.4f}")
        scheduler.step()

    # ── Résultats ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Rapport de classification :")
    print(classification_report(all_targets, all_preds, target_names=train_df['emotion'].unique()))

    plot_history(history)
    plot_confusion_matrix(all_targets, all_preds, train_df['emotion'].unique())


if __name__ == "__main__":
    main()

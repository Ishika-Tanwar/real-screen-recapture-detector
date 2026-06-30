"""
train.py — Fine-tune MobileNetV2 to detect screen recaptures.

Folder structure expected:
    data/
        real/     ← photos of real things
        screen/   ← photos of screens

Usage:
    python train.py

Outputs:
    model.pth   ← saved model weights (used by predict.py)
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR   = "data"          # folder containing real/ and screen/ subfolders
MODEL_OUT  = "model.pth"     # where to save the trained weights
IMG_SIZE   = 224            # MobileNetV2 expects 224×224
BATCH_SIZE = 16
EPOCHS     = 25
LR         = 1e-3            # low LR since we're fine-tuning
VAL_SPLIT  = 0.2             # 20% of data used for validation
SEED       = 42

# ── Device ────────────────────────────────────────────────────────────────────
device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()   # Apple Silicon
    else "cpu"
)
print(f"Using device: {device}")

# ── Transforms ────────────────────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── Dataset ───────────────────────────────────────────────────────────────────
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)

print(f"Classes found: {full_dataset.class_to_idx}")
assert "real" in full_dataset.class_to_idx, "Expected a 'real' subfolder in data/"
assert "screen" in full_dataset.class_to_idx, "Expected a 'screen' subfolder in data/"
SCREEN_IDX = full_dataset.class_to_idx["screen"]

n_val   = int(len(full_dataset) * VAL_SPLIT)
n_train = len(full_dataset) - n_val
train_set, val_set = random_split(
    full_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(SEED)
)

val_set.dataset = datasets.ImageFolder(DATA_DIR, transform=val_tf)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Training samples: {n_train}  |  Validation samples: {n_val}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

for param in model.features.parameters():
    param.requires_grad = False

for param in model.features[-3:].parameters():
    param.requires_grad = True

in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(in_features, 1),
)

model = model.to(device)

# ── Training ──────────────────────────────────────────────────────────────────
criterion = nn.BCEWithLogitsLoss()
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

best_val_acc = 0.0
patience = 6
epochs_no_improve = 0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, train_correct = 0.0, 0

    for images, labels in train_loader:
        images = images.to(device)
        targets = (labels == SCREEN_IDX).float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss   = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        train_loss    += loss.item() * images.size(0)
        preds          = (torch.sigmoid(logits) > 0.5).float()
        train_correct += (preds == targets).sum().item()

    scheduler.step()

    model.eval()
    val_loss, val_correct = 0.0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images  = images.to(device)
            targets = (labels == SCREEN_IDX).float().unsqueeze(1).to(device)

            logits = model(images)
            loss   = criterion(logits, targets)

            val_loss    += loss.item() * images.size(0)
            preds        = (torch.sigmoid(logits) > 0.5).float()
            val_correct += (preds == targets).sum().item()

    train_acc = train_correct / n_train
    val_acc   = val_correct   / n_val

    print(f"Epoch {epoch:02d}/{EPOCHS}  "
          f"train_loss={train_loss/n_train:.4f}  train_acc={train_acc:.3f}  "
          f"val_loss={val_loss/n_val:.4f}  val_acc={val_acc:.3f}")

    # Save the best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_OUT)
        print(f"  ✓ Saved new best model (val_acc={val_acc:.3f})")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"\nNo improvement for {patience} epochs, stopping early.")
            break
print(f"\nDone! Best validation accuracy: {best_val_acc:.3f}")
print(f"Model saved to: {MODEL_OUT}")
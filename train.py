import torch
from torch import nn, optim
from sklearn.metrics import f1_score, accuracy_score
from model import GESTURE_NET
from data import load_dataset, get_dataloader
from logger import CompleteLogger

# Config
EPOCHS = 25
BATCH_SIZE = 1
LR = 0.00007
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Logger
logger = CompleteLogger("logs", "train", experiment_name="GESTURE-Net")
logger.logger.info("Training Started")

# Data
train_files = load_dataset("./features/train")
val_files = load_dataset("./features/val")
train_loader = get_dataloader(train_files, batch_size=BATCH_SIZE)
val_loader = get_dataloader(val_files, batch_size=BATCH_SIZE)

# Model
model = GESTURE_NET().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
criterion = nn.BCEWithLogitsLoss()

# Training Loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []
    for features, gestures, labels in train_loader:
        features, gestures, labels = features.to(DEVICE), gestures.to(DEVICE), labels.to(DEVICE).float()
        outputs = model(features, gestures)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).int()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    logger.logger.info(f"Epoch {epoch+1}/{EPOCHS}, Loss={total_loss:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

# Save model
torch.save(model.state_dict(), "gesture_net.pth")
logger.close()

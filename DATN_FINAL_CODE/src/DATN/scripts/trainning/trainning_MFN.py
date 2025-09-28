
# ======================================================================
# FILE:         train_mfn.py
# MÔ TẢ:        Huấn luyện MobileFaceNet với train/val, lưu checkpoint,
#               lưu CSV metrics và biểu đồ riêng, Train+Val Loss gộp
# ======================================================================

import os

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import csv
from src.DATN.model.network.network_MFN import MobileFaceNet
from src.DATN.model.loss.loss_mobilefacenet import ArcFaceLoss
from src.DATN.config.cau_hinh import CONFIG

# ================== PATH ==================
MFN_TRAIN       = CONFIG["DATA"]["MFN_TRAIN"]
MFN_VAL         = CONFIG["DATA"]["MFN_VAL"]
SAVE_DIR        = CONFIG["MODEL"]["MFN_MODEL_PATH"]
MFN_MODEL_PATH = os.path.join(SAVE_DIR, "Mobilefacene_train.pt")
LOG             = os.path.join(SAVE_DIR, 'log')
WEIGHT        = os.path.join(SAVE_DIR, 'Weights')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG, exist_ok=True)
os.makedirs(WEIGHT, exist_ok=True)

# ================== TRAIN CONFIG ==================
BATCH_SIZE      = CONFIG["TRAIN"]["BATCH_SIZE"]
EPOCHS          = CONFIG["TRAIN"]["EPOCHS"]
LR              = CONFIG["TRAIN"]["LR"]
END_LR          = 0.000001
EMBEDDING_SIZE  = CONFIG["TRAIN"]["EMBEDDING_SIZE"]
DEVICE          = CONFIG["TRAIN"]["DEVICE"]
transform       = CONFIG["TRAIN"]["TRANSFORM"]

# ================== DATASET & DATALOADER ==================
train_dataset = datasets.ImageFolder(root=MFN_TRAIN, transform=transform)
val_dataset   = datasets.ImageFolder(root=MFN_VAL , transform=transform)

#gansn nhãn  với mục con là 1 class
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True) # drop_last loại bỏ batch last nếu < batch size
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ================== MODEL ==================
# load model
model = MobileFaceNet()
model.load_state_dict(torch.load(MFN_MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
# ================== LOSS & OPTIMIZER ==================
criterion = ArcFaceLoss(
    embedding_size=EMBEDDING_SIZE,
    num_classes=len(train_dataset.classes),
    s=64.0,
    m=0.5
).to(DEVICE)

optimizer = optim.Adam(list(model.parameters()) + [criterion.weight], lr=LR)
# Scheduler ví dụ CosineAnnealingLR với LR cuối là endLR
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=END_LR)
# ================== TRACKING ==================
train_losses = []
val_losses   = []
val_accs    = []
val_precisions = []
val_recalls    = []
best_val_acc = 0.0              # độ chính xác
best_model_state = None         # biến lưu tạm best.pt
saved_checkpoints = set()       # set lưu danh sách các file .pt cần lưu

# ================== CSV ==================
csv_path = os.path.join(LOG, "mfn_metrics.csv")
if not os.path.exists(csv_path):
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Val_Accuracy", "Val_Precision", "Val_Recall"])


# ================== TRAIN LOOP ==================
for epoch in range(1, EPOCHS + 1):
    # --- TRAIN ---
    model.train()
    total_train_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        embeddings = model(imgs)
        loss = criterion(embeddings, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Cập nhật LR
        scheduler.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # --- VALIDATION ---
    model.eval()
    total_val_loss = 0
    all_labels = []
    all_preds  = []

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            embeddings = model(imgs)
            loss = criterion(embeddings, labels)
            total_val_loss += loss.item()
            # tính class dự đoán
            logits = torch.matmul(embeddings, criterion.weight.t())
            preds = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    val_acc  = accuracy_score(all_labels, all_preds)
    val_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    val_rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    val_accs.append(val_acc)
    val_precisions.append(val_prec)
    val_recalls.append(val_rec)

    print(f"Epoch [{epoch}/{EPOCHS}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    # --- Lưu CSV ---
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, avg_train_loss, avg_val_loss, val_acc, val_prec, val_rec])

    # --- Checkpoint ---
    # Nếu acc tốt hơn thì cập nhật model vào biến nhớ
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()  # lưu state_dict vào RAM
        counter = 0
        print(f" Epoch trả về Acc có giá trị tốt nhất hiện tại là :{epoch} với Val Acc = {best_val_acc:.4f}")
    else:
        counter = counter + 1       # early stopping  ( neu sau n counter model ko cai thien thi thoat)
    # Lưu checkpoint định kỳ (5 epoch 1 lần)
    if epoch % 5 == 0:
        path = os.path.join(WEIGHT, f"mfn_epoch{epoch}.pt")
        torch.save(model.state_dict(), path)
        print(f"Saved checkpoint: {path}")

    # Lưu checkpoint cuối cùng
    if epoch == EPOCHS or counter >= 10:
        path = os.path.join(WEIGHT, "mfn_last.pt")
        torch.save(model.state_dict(), path)
        print(f"Saved checkpoint: {path}")
    if counter >= 10:
        print(f"Early stopping triggered at epoch {epoch}.")
        EPOCHS = epoch
        break

# Sau training, mới lưu best
if best_model_state is not None:
    best_path = os.path.join(SAVE_DIR, "mfn_best.pt")
    torch.save(best_model_state, best_path)
    print(f"Saved best checkpoint (Val Acc: {best_val_acc:.4f}): {best_path}")
# --- VẼ VÀ LƯU BIỂU ĐỒ ---
# Doc file csv
csv_path = os.path.join(LOG, "mfn_metrics.csv")
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print("Khong tim thay file csv")
# Train + Val Loss
plt.figure(figsize=(10, 8))
plt.plot(range(1, EPOCHS + 1), df["Train_Loss"], 'b-o', label='Train Loss')
plt.plot(range(1, EPOCHS + 1), df["Val_Loss"], 'r-o', label='Val Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train & Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(LOG, "train_val_loss.png"))
plt.show()

# Val Accuracy
plt.figure(figsize=(10, 8))
plt.plot(range(1, EPOCHS  + 1), df["Val_Accuracy"], 'g-o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy")
plt.grid(True)
plt.savefig(os.path.join(LOG, "val_accuracy.png"))
plt.show()

# Val Precision
plt.figure(figsize=(10, 8))
plt.plot(range(1, EPOCHS  + 1), df["Val_Precision"], 'm-o')
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.title("Validation Precision")
plt.grid(True)
plt.savefig(os.path.join(LOG, "val_precision.png"))
plt.show()

# Val Recall
plt.figure(figsize=(10, 8))
plt.plot(range(1, EPOCHS  + 1), df["Val_Recall"], 'c-o')
plt.xlabel("Epoch")
plt.ylabel("Recall")
plt.title("Validation Recall")
plt.grid(True)
plt.savefig(os.path.join(LOG, "val_recall.png"))
plt.show()

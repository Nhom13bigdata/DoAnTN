# ======================================================================
# FILE:         predict_flw.py
# MÔ TẢ:        Đánh giá MobileFaceNet + FAISS theo VERIFICATION
#                 + Open-set (FAISS, FAR/TRR)
#                 + Verification (Cosine similarity: AUC, EER)
# ======================================================================

import os
import cv2
import torch
import random
import faiss
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from src.DATN.config.cau_hinh import CONFIG
from src.DATN.model.network.network_MFN import MobileFaceNet
from sklearn.metrics import roc_curve, auc
import torch.nn.functional as F

# ================= CONFIG ==============
MFN_TEST         = CONFIG["DATA"]["MFN_TEST"]         # Thư mục test
FAISS_INDEX_PATH = CONFIG["DATA"]["FAISS_INDEX_PATH"] # FAISS index đã build
DEVICE           = CONFIG["TRAIN"]["DEVICE"]
EMBEDDING_SIZE   = CONFIG["TRAIN"]["EMBEDDING_SIZE"]
SAVE_DIR         = CONFIG["MODEL"]["MFN_MODEL_PATH"]
transfrom        =  CONFIG["TRAIN"]["TRANSFORM"]
predict_MFN      = os.path.join(SAVE_DIR, "predict_MFN")

# ================= Transform ảnh (resize + normalize) =================
transform = transfrom

os.makedirs(predict_MFN, exist_ok=True)

# ================= Load MobileFaceNet =================
MFN_path = os.path.join(CONFIG["MODEL"]["MFN_MODEL_PATH"], "mfn_best.pt")
MFN_model = MobileFaceNet()
MFN_model.load_state_dict(torch.load(MFN_path, map_location=DEVICE))
MFN_model.to(DEVICE)
MFN_model.eval()

# ================= Load FAISS Index =================
index = faiss.read_index(FAISS_INDEX_PATH)

# ================= Hàm trích xuất embedding từ ảnh =================
def get_embedding(img_path):
    """
    Input: đường dẫn ảnh
    Output: vector embedding đã L2-normalized (norm = 1)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = transform(img).to(DEVICE)
    img = img.unsqueeze(0)
    with torch.no_grad():
        emb = MFN_model(img)                   # trích xuất embedding
        emb = F.normalize(emb, p=2, dim=1)     # L2-normalize (norm = 1)
        emb = emb.cpu().numpy().astype('float32').reshape(-1)
    return emb

# ================= FAISS Prediction (open-set) =================
def predict_faiss(embedding, threshold=1.0):
    """
    Input: embedding (vector)
    Output: (id, dist)
      - Nếu dist >= threshold → Unknown (-1)
      - Nếu dist < threshold → Known (id trong index)
    """
    emb = embedding.reshape(1, -1)
    D, I = index.search(emb, 1)   # tìm nearest neighbor
    dist, idx = D[0][0], I[0][0]
    if dist >= threshold:
        return -1, dist  # Unknown
    else:
        return idx, dist  # Known

# ================= Tính EER từ FPR, TPR =================
def compute_eer(fpr, tpr, thresholds):
    """
    EER (Equal Error Rate): điểm mà FAR = FRR
    FAR = fpr
    FRR = 1 - tpr
    """
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx_eer = np.argmin(abs_diffs)   # vị trí sai khác nhỏ nhất
    eer = (fpr[idx_eer] + fnr[idx_eer]) / 2
    eer_threshold = thresholds[idx_eer]
    return eer, eer_threshold

# ================= Evaluation =================
def evaluate(test_root=MFN_TEST, threshold=0.5):
    """
    Đánh giá gồm 2 phần:
      - FAISS open-set: tính FAR, TRR
      - Cosine verification: tính AUC, EER
    """
    faiss_labels, faiss_dists = [], []
    cos_labels, cos_scores = [], []

    # ================= Duyệt từng lớp (mỗi người) =================
    classes = [c for c in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, c))]

    for class_name in classes:
        class_path = os.path.join(test_root, class_name)
        imgs = [os.path.join(class_path, f)
                for f in os.listdir(class_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        embeddings = []
        for img_path in tqdm(imgs, desc=f"Processing {class_name}"):
            try:
                emb = get_embedding(img_path)
                embeddings.append(emb)
            except Exception as e:
                print(f"Bỏ qua {img_path}: {e}")
                continue

            # ---------- FAISS open-set ----------
            pred_label, dist = predict_faiss(emb, threshold=threshold)
            if pred_label == -1:
                faiss_labels.append(1)  # Nhận đúng Unknown
            else:
                faiss_labels.append(0)  # Nhận nhầm Known
            faiss_dists.append(dist)

        # ---------- Verification: Same person ----------
        same_pairs = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                emb_i, emb_j = embeddings[i], embeddings[j]
                sim = np.dot(emb_i, emb_j)   # cosine similarity (đã normalize)
                cos_scores.append(float(sim))
                cos_labels.append(1)  # same person
                same_pairs.append((emb_i, emb_j))
        # Tinh so luong anh cho moi other_person de cap voi same person
        num_same_pairs = len(same_pairs)
        num_other_classes = len(classes) - 1
        pairs_per_other = max(1, num_same_pairs // num_other_classes)  # số cặp cho mỗi người khác
        # ---------- Verification: Different person ----------
        diff_pairs = []
        for other_class in classes:
            if other_class == class_name:
                continue
            other_path = os.path.join(test_root, other_class)
            other_imgs = [os.path.join(other_path, f)
                          for f in os.listdir(other_path)
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            if not other_imgs:
                continue
            # chon so luong anh can cap cua moi other person
            sampled_imgs = random.sample(other_imgs, min(pairs_per_other, len(other_imgs)))

            for img_path in sampled_imgs:
                try:
                    emb_other = get_embedding(img_path)
                except Exception as e:
                    print(f"Bỏ qua {img_path}: {e}")
                    continue

                emb_i = random.choice(embeddings)  # chỉ 1 embedding, không lặp tất cả
                diff_pairs.append((emb_i, emb_other))
    # Tính cosine similarity và gán nhãn
    for emb_i, emb_j in diff_pairs:
        sim = np.dot(emb_i, emb_j)
        cos_scores.append(float(sim))
        cos_labels.append(0)  # different person
    # ================= Kết quả FAISS open-set =================
    faiss_labels = np.array(faiss_labels)
    total_faiss = len(faiss_labels)
    true_reject = np.sum(faiss_labels == 1)
    false_accept = np.sum(faiss_labels == 0)
    FAR = false_accept / total_faiss
    TRR = true_reject / total_faiss

    print("\n===== FAISS Open-set Evaluation =====")
    print(f"Total Samples   : {total_faiss}")
    print(f"True Reject     : {true_reject}")
    print(f"False Accept    : {false_accept}")
    print(f"FAR (False Accept Rate): {FAR:.4f}")
    print(f"TRR (True Reject Rate) : {TRR:.4f}")

    # ================= Verification: Cosine similarity =================
    cos_labels = np.array(cos_labels)
    cos_scores = np.array(cos_scores)
    fpr, tpr, thresholds = roc_curve(cos_labels, cos_scores)
    roc_auc = auc(fpr, tpr)
    eer, eer_threshold = compute_eer(fpr, tpr, thresholds)

    print("\n===== Cosine Similarity Evaluation (Verification) =====")
    print(f"Number of pairs: {len(cos_labels)}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"EER: {eer:.4f} tại threshold = {eer_threshold:.4f}")

    # ================= Vẽ biểu đồ =================
    # --- FAISS open-set ---
    plt.figure(figsize=(7,5))
    plt.bar(["FAISS True Reject", "FAISS False Accept"],
            [true_reject, false_accept], color=["green", "red"])
    plt.title("FAISS Open-set Evaluation Counts")
    plt.ylabel("Number of Samples")
    plt.savefig(os.path.join(predict_MFN, "faiss_eval.png"))
    plt.show()

    # --- ROC Curve (cosine) ---
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.axvline(x=eer, color="red", linestyle="--", label=f"EER = {eer:.4f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Same/Different Person (Cosine)")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(predict_MFN, "cosine_roc.png"))
    plt.show()

if __name__ == "__main__":
    evaluate()

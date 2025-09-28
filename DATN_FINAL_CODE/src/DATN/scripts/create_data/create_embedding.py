# ======================================================================
# FILE:         create_embeddings_faiss.py
# MÔ TẢ:        Tạo embeddings từ dataset đã crop và lưu vào FAISS
#               Mỗi class (đối tượng) có một ID cố định
#               Tất cả embedding cùng class dùng chung ID
#               Lưu mapping class_id -> tên user ngoài FAISS
# ======================================================================

import os
import cv2
import torch.nn.functional as F
import torch
import faiss
import numpy as np
from torchvision import transforms
from PIL import Image
from  src.DATN.model.network.network_MFN import MobileFaceNet
from src.DATN.config.cau_hinh import CONFIG

# ===== CONFIG =====
DEVICE             = CONFIG["TRAIN"]["DEVICE"]
EMBEDDING_SIZE     = CONFIG["TRAIN"]["EMBEDDING_SIZE"] # kích thước embedding (128)
IMG_SIZE           = CONFIG["IMAGE"]["IMG_SIZE_PROCESSED"] # kích thước input (112x112)
MFN_TRAIN          = CONFIG["DATA"]["MFN_TRAIN"] # input
FAISS_INDEX_PATH   = CONFIG["DATA"]["FAISS_INDEX_PATH"] # faiss lưu embedding với class tương ứng
CLASS_MAPPING_PATH = CONFIG["DATA"]["MAP_ID"]  # class_id -> user_name

MFN_MODEL_PATH    = CONFIG["MODEL"]["MFN_MODEL_PATH"] # file .pt chứa tham số của mô hình

# ===== Transform =====
transform =  CONFIG["TRAIN"]["TRANSFORM"]


# ===== Load MobileFaceNet =====
MFN_path = os.path.join(MFN_MODEL_PATH, "mfn_best.pt")
model = MobileFaceNet()
# 1. Load trọng số vào model
state_dict = torch.load(MFN_path, map_location=DEVICE)
model.load_state_dict(state_dict)  # load vào module

# 2. Chuyển model sang device
model = model.to(DEVICE)
model.eval()  # nếu chỉ inference

# ===== Tạo FAISS Index với IDMap =====
index = faiss.IndexFlatL2(EMBEDDING_SIZE)   # dùng khoảng cách L2
index = faiss.IndexIDMap(index)             # mỗi vector có ID (class_id)
class_mapping = {}                          # lưu class_id -> user_name

# ===== Duyệt dataset =====
# Giả sử mỗi thư mục con trong PROCESSED_DIR là một user/class
for class_id, class_name in enumerate(sorted(os.listdir(MFN_TRAIN)), start=201):
    # kHỞI TẠO LABEL( ID)
    class_folder = os.path.join(MFN_TRAIN, class_name)
    if not os.path.isdir(class_folder):
        continue

    embeddings_list = []

    for file_name in os.listdir(class_folder):
        if not file_name.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(class_folder, file_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # đổi sang RGB
        img = Image.fromarray(img)  # numpy -> PIL
        img_tensor = transform(img).unsqueeze(0).to(DEVICE) # unsqueeze(0) thêm 1 chiều vào ví tri đầu ví (4,) -> [1,4]

        with torch.no_grad():
            emb_tensor = model(img_tensor)                   # Lấy embedding từ model (tensor) shape = (1, 128)
            emb_tensor = F.normalize(emb_tensor, p=2, dim=1) # Chuẩn hóa L2 trực tiếp trên tensor
            emb = emb_tensor.cpu().numpy().astype('float32') # Chuyển sang numpy để lưu hoặc add vào FAISS
            embeddings_list.append(emb)

    if embeddings_list:
        embeddings_array = np.vstack(embeddings_list) # tao thanh ma tran 2 chieu [1, 2] [3, 4] -> [[1 ,2 ], [3, 4]]
        ids_array = np.full((embeddings_array.shape[0],), class_id, dtype=np.int64) # tạo id tương ứng với từng embedding
        index.add_with_ids(embeddings_array, ids_array)
        class_mapping[class_id] = class_name

# ===== Lưu FAISS Index =====
faiss.write_index(index, FAISS_INDEX_PATH)

# ===== Lưu class mapping ra file =====
with open(CLASS_MAPPING_PATH, "w", encoding="utf-8") as f:
    for cid, cname in class_mapping.items():
        f.write(f"{cid}\t{cname}\n")
print(f"FAISS index saved to {FAISS_INDEX_PATH}")


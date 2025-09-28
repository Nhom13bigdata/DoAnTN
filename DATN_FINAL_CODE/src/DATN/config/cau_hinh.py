# ======================================================================
# FILE:         cau_hinh.py
# MÔ TẢ:        Cấu hình đường dẫn, tham số train, model, image cho dự án Face Recognition
# ======================================================================

import os
import torch
from torchvision import transforms
# ================== BASE PATH ==================
ROOT_DIR   = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")) # tuyet doi
DATA_DIR   = os.path.join(ROOT_DIR, "Data")
SRC_DIR    = os.path.join(ROOT_DIR, "src")
DEPLOY_DIR = os.path.join(ROOT_DIR, "Deploy")
WEIGHT_DIR = os.path.join(SRC_DIR, "DATN", "Weights")
IMG_USER_DB = os.path.join(DATA_DIR, "IMG_USER_DB")
# ================== Data PATH ==================
# thư mục huân luyê YOLO
WIDER_FACE_DIR      = os.path.join(DATA_DIR, "WIDER FACE")             # bộ WIDER_FACE_DIR, chứa img/ và label/
WIDER_FACE_IMG    = os.path.join(WIDER_FACE_DIR, "images")             # /images
WIDER_FACE_LABEL    = os.path.join(WIDER_FACE_DIR, "labels")           # /labels

# Thư mục raw ( chưa xử lý)
VGGFACE_DIR     = os.path.join(DATA_DIR, "VGGFACE")                    # bộ GGFACE, chứa train/ và test/
# Thư mục ảnh gốc train/test
VGGFACE_TRAIN   = os.path.join(VGGFACE_DIR, "train")                   # /train
VGGFACE_TEST    = os.path.join(VGGFACE_DIR, "test")                     # /test

# Thư mục YOLO( thư mục dùng cho YOLO)
YOLO_DIR        = os.path.join(DATA_DIR, "YOLO data")                   # /YOLO data
# Thư mục ảnh và nhãn
YOLO_IMGS       = os.path.join(YOLO_DIR, "images")                      # thư mục chứa /img
YOLO_LABELS     = os.path.join(YOLO_DIR, "labels")                      # thư mục chứa /label
# Thư mục ảnh train/val/test
IMGS_TRAIN      = os.path.join(YOLO_IMGS, "train")                      # img/train
IMGS_VAL        = os.path.join(YOLO_IMGS, "val")                        # img/val
IMGS_TEST       = os.path.join(YOLO_IMGS, "test")                       # img/test
# Thư mục nhãn train/val/test
LABELS_TRAIN    = os.path.join(YOLO_LABELS, "train")                    # label/train
LABELS_VAL      = os.path.join(YOLO_LABELS, "val")                      # label/val
LABELS_TEST     = os.path.join(YOLO_LABELS, "test")                     # label/test

# Thư mục MFN data ( thư mục dùng cho MFN)
MFN_DIR           = os.path.join(DATA_DIR, "MFN data")                  # /MFN data
# Thư mục train/val/test
MFN_TRAIN       = os.path.join(MFN_DIR, "train")                        # /train
MFN_VAL         = os.path.join(MFN_DIR, "val")                          # /val
MFN_TEST        = os.path.join(MFN_DIR, "test")                         # test

# Thư mục embedding / database
EMBEDDING_DIR      = os.path.join(DATA_DIR, "Embedding and DB")         # EMBEDDING AND DB
FAISS_INDEX_PATH   = os.path.join(EMBEDDING_DIR, "faiss_index.index")   # /faiss_index
DB_PATH            = os.path.join(EMBEDDING_DIR, "users_info.db")       # /users_info.db
MAP_ID             = os.path.join(EMBEDDING_DIR, "map_id.txt")          # /map_id.txt

# ================== TRAINING CONFIG ==================
BATCH_SIZE      = 16                                                      # So luong img moi lan vao model
EPOCHS          = 50                                                     # so vong lap
LEARNING_RATE   = 0.001                                                  # tham so toc do hoc
EMBEDDING_SIZE  = 128                                                    # so chieu embedding
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"         # Neu co GPU thi dung ko thi CPU
YOLO_YAML       =  os.path.join(SRC_DIR, "DATN", "config", "YOLO.yaml")  # file thong tin can de train YOLO
#YOLO_PREDICT_YAML    =  os.path.join(SRC_DIR, "DATN", "config", "predict_YOLO.yaml")  # file thong tin can de predict YOLO
# ===== Dataset & Transform =====
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # chỉ tính khi đàu vào là tensor
])
# ================== MODEL PATH ==================
YOLO_MODEL_PATH      = os.path.join(WEIGHT_DIR, "YOLO Weight")                          # đường dẫn luu tham số YOLO
YOLO_FACE            = os.path.join(YOLO_MODEL_PATH, "yolov8m-face.pt")         # file .pt truoc khi fine_tune YOLO
MFN_MODEL_PATH       = os.path.join(WEIGHT_DIR, "MFN Weight")                           # duong dan luu tham số MFN
YOLO_CONF            = 0.6     # Confidence threshold YOLO                              # do tin cay

# ================== IMAGE CONFIG ==================
IMG_SIZE_PROCESSED  = (112, 112)    # Kích thước ảnh sau khi crop/process
YOLO_INPUT_SIZE     = (320, 320)    # Input size YOLO detect
# Số lượng ảnh tốt thiểu của từng người
MIN_IMAGES_PER_USER = 70   # số ảnh tối thiểu mỗi user
MAX_IMAGES_PER_USER = 110   # số ảnh tối đa mỗi user

# Tỷ lệ chia train/test/val
TRAIN_RATIO      = 0.7         # Tỷ lệ ảnh train
VAL_RATIO        = 0.2          # Tỷ lệ ảnh val
TEST_RATIO       = 0.1         # tỷ lệ ảnh test
# Số lượng đuối tượng
USER_TRAIN   = 100                  # Số đối tượng  train
USER_VAL     = 100                 # Số đối tượng  val
USER_TEST    = 50                   # Số đối tượng  test
# Điều kiện lọc
MIN_SIZE            = 150                 # size tối thiểu của ảnh
BRIGHTNESS          = (60, 170)          # Ngưỡng độ sáng (mean grayscale)
IMG_EXTS            = [".jpg", ".png"]  # đuôi file


# ================== GROUPED CONFIG ==================
CONFIG = {
    "DATA": {
        "ROOT" : ROOT_DIR,
        "DATA_DIR": DATA_DIR,
        "VGGFACE_DIR": VGGFACE_DIR ,
        "VGGFACE_TRAIN": VGGFACE_TRAIN,
        "VGGFACE_TEST": VGGFACE_TEST,
        "WIDER_FACE_DIR": WIDER_FACE_DIR,
        "WIDER_FACE_LABEL": WIDER_FACE_LABEL,
        "WIDER_FACE_IMG": WIDER_FACE_IMG,
        "YOLO_DIR": YOLO_DIR,
        "YOLO_IMGS": YOLO_IMGS ,
        "YOLO_LABELS": YOLO_LABELS,
        "IMGS_TRAIN": IMGS_TRAIN ,
        "IMGS_VAL": IMGS_VAL,
        "IMGS_TEST": IMGS_TEST,
        "LABELS_TRAIN": LABELS_TRAIN ,
        "LABELS_VAL": LABELS_VAL,
        "LABELS_TEST": LABELS_TEST,
        "MFN_DIR": MFN_DIR,
        "MFN_TRAIN": MFN_TRAIN ,
        "MFN_VAL": MFN_VAL,
        "MFN_TEST": MFN_TEST,
        "EMBEDDING_DIR" : EMBEDDING_DIR,
        "FAISS_INDEX_PATH": FAISS_INDEX_PATH,
        "DB_PATH": DB_PATH,
        "IMG_USER_DB": IMG_USER_DB,
        "MAP_ID": MAP_ID,
    },
    "TRAIN": {
        "BATCH_SIZE": BATCH_SIZE,
        "EPOCHS": EPOCHS,
        "LR": LEARNING_RATE,
        "EMBEDDING_SIZE": EMBEDDING_SIZE,
        "DEVICE": DEVICE,
        "YOLO_YAML": YOLO_YAML,
        "TRANSFORM" : transform,
    },
    "MODEL": {
        "WEIGHT_DIR": WEIGHT_DIR,
        "MFN_MODEL_PATH": MFN_MODEL_PATH ,
        "YOLO_MODEL_PATH": YOLO_MODEL_PATH,
        "YOLO_FACE" : YOLO_FACE,
        "YOLO_CONF": YOLO_CONF,
    },
    "IMAGE": {
        "IMG_SIZE_PROCESSED": IMG_SIZE_PROCESSED,
        "YOLO_INPUT_SIZE": YOLO_INPUT_SIZE,
        "MIN_IMAGES_PER_USER": MIN_IMAGES_PER_USER,
        "MAX_IMAGES_PER_USER": MAX_IMAGES_PER_USER,
        "USER_TRAIN": USER_TRAIN,
        "USER_VAL": USER_VAL ,
        "USER_TEST": USER_TEST  ,
        "MIN_SIZE": MIN_SIZE,
         "BRIGHTNESS" : BRIGHTNESS,
        "IMG_EXTS": IMG_EXTS,
        "TRAIN_RATIO": TRAIN_RATIO,
        "VAL_RATIO": VAL_RATIO ,
        "TEST_RATIO" : TEST_RATIO,
    },

}

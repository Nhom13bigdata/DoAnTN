# ======================================================================
# FILE:         processing_YOLO.py
# MÔ TẢ:        Lọc ảnh và chia train/val/test, copy label
# ======================================================================

import os
import cv2
import shutil
import random
from tqdm import tqdm
from src.DATN.config.cau_hinh import CONFIG

# ================== PATH CONFIG ==================
WIDER_FACE_DIR= CONFIG["DATA"]["WIDER_FACE_DIR"]        # bộ WIDER_FACE_DIR, chứa img/ và label/
IMG_DIR   =    CONFIG["DATA"]["WIDER_FACE_IMG"]         # muc /img
LABEL_DIR =    CONFIG["DATA"]["WIDER_FACE_LABEL"]       # muc /label

YOLO_IMGS    = CONFIG["DATA"]["YOLO_IMGS"]              # ảnh YOLO data/YOLO
IMGS_TRAIN   = CONFIG["DATA"]["IMGS_TRAIN"]             # ảnh /train
IMGS_VAL     = CONFIG["DATA"]["IMGS_VAL"]               # ảnh /val
IMGS_TEST    = CONFIG["DATA"]["IMGS_TEST"]              # ảnh /test

YOLO_LABELS  = CONFIG["DATA"]["YOLO_LABELS"]            # YOLO datalabel YOLO
LABELS_TRAIN = CONFIG["DATA"]["LABELS_TRAIN"]           # label /train
LABELS_VAL   = CONFIG["DATA"]["LABELS_VAL"]             # label /val
LABELS_TEST  = CONFIG["DATA"]["LABELS_TEST"]            # label /test

IMG_EXTS   = tuple(CONFIG["IMAGE"]["IMG_EXTS"])         # định dạng ảnh hợp lệ (.jpg)
MIN_SIZE   = CONFIG["IMAGE"]["MIN_SIZE"]                # kích thước ảnh tối thiểu
BRIGHTNESS = CONFIG["IMAGE"]["BRIGHTNESS"]              # độ sáng khi augment

TRAIN_RATIO = CONFIG["IMAGE"]["TRAIN_RATIO"]            # tỷ lệ train
VAL_RATIO   = CONFIG["IMAGE"]["VAL_RATIO"]              # tỷ lệ val
TEST_RATIO  = CONFIG["IMAGE"]["TEST_RATIO"]             # tỷ lệ test

CLASS_FACE = "0"                               # class khuôn mặt

# ================== CHECK IMAGE VALIDITY ==================
def check_image_valid(path):
    try:
        img = cv2.imread(path)
        if img is None:
            return False
        h, w = img.shape[:2]
        if min(h, w) < MIN_SIZE:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = gray.mean()
        if brightness < BRIGHTNESS[0] or brightness > BRIGHTNESS[1]:
            return False
        return True
    except:
        return False

# ================== CHECK LABEL FACE ==================
# def check_label_single_person(label_path, class_face=CLASS_FACE):
#     try:
#         with open(label_path, "r") as f:
#             lines = [line.strip() for line in f.readlines() if line.strip()]
#         if len(lines) != 1:
#             return False
#         return lines[0].startswith(class_face)
#     except:
#         return False
def check_label_all_faces(label_path, class_face=CLASS_FACE, max_people=2):
    """
    Kiểm tra file nhãn:
    - Chỉ chứa các bounding box của class_face (ví dụ: '0' nếu face là class 0).
    - Cho phép nhiều dòng (nhiều khuôn mặt) nhưng tối đa `max_people`.
    """
    try:
        with open(label_path, "r") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:  # không có nhãn nào
            return False

        # Nếu số người > max_people → False
        if len(lines) > max_people:
            return False

        # Kiểm tra tất cả dòng đều bắt đầu bằng class_face
        return all(line.startswith(str(class_face)) for line in lines)

    except Exception as e:
        print(f"Lỗi khi đọc {label_path}: {e}")
        return False

# ================== SPLIT DATASET ==================
def split_dataset(image_list, label_list):
    combined = list(zip(image_list, label_list))
    random.shuffle(combined)
    image_list, label_list = zip(*combined)

    total = len(image_list)
    n_train = int(total * TRAIN_RATIO)
    n_val   = int(total * VAL_RATIO)

    train_images = image_list[:n_train]
    train_labels = label_list[:n_train]

    val_images = image_list[n_train:n_train+n_val]
    val_labels = label_list[n_train:n_train+n_val]

    test_images = image_list[n_train+n_val:]
    test_labels = label_list[n_train+n_val:]

    return (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)

# ================== COPY IMAGES + LABELS ==================
def copy_images_labels(image_list, label_list, dest_images, dest_labels):
    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_labels, exist_ok=True)
    for img_file, label_file in tqdm(zip(image_list, label_list), total=len(image_list), desc=f"Sao chép vào {dest_images}"):
        shutil.copy(os.path.join(IMG_DIR, img_file), os.path.join(dest_images, img_file))
        shutil.copy(os.path.join(LABEL_DIR, label_file), os.path.join(dest_labels, label_file))

# ================== PROCESS DATASET ==================
def process_dataset():
    # Lấy danh sách ảnh
    image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(IMG_EXTS)]
    valid_images = []
    valid_labels = []

    for img_file in tqdm(image_files, desc="Lọc ảnh hợp lệ"):
        label_file = os.path.splitext(img_file)[0] + ".txt"
        img_path = os.path.join(IMG_DIR, img_file)
        label_path = os.path.join(LABEL_DIR, label_file)

        if (os.path.exists(label_path)
            and check_image_valid(img_path)
            and check_label_all_faces(label_path)):
            valid_images.append(img_file)
            valid_labels.append(label_file)

    print(f"Tổng ảnh hợp lệ: {len(valid_images)}")

    # Chia train/val/test
    (train_images, train_labels), (val_images, val_labels), (test_images, test_labels)= split_dataset(valid_images, valid_labels)

    # Copy ảnh và label
    copy_images_labels(train_images, train_labels, IMGS_TRAIN, LABELS_TRAIN)
    copy_images_labels(val_images, val_labels, IMGS_VAL, LABELS_VAL)
    copy_images_labels(test_images, test_labels, IMGS_TEST, LABELS_TEST)

# ================== MAIN ==================
if __name__ == "__main__":
    process_dataset()
    print("Hoàn tất xử lý dataset!")

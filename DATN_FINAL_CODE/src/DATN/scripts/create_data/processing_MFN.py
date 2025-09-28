# ======================================================================
# FILE:         preprocess_mfn.py
# MÔ TẢ:        Tiền xử lý dữ liệu cho MFN: lọc ảnh, detect face bằng YOLO,
#               crop & resize, chia train/val/test
# ======================================================================

import os
import cv2
import numpy as np
import shutil
from tqdm import tqdm
from ultralytics import YOLO
from src.DATN.config.cau_hinh import CONFIG

# ================== Load config ==================
# input
VGGFACE_TRAIN      = CONFIG["DATA"]["VGGFACE_TRAIN"]            # /train
VGGFACE_TEST       = CONFIG["DATA"]["VGGFACE_TEST"]             # /test
# output
MFN_TRAIN       = CONFIG["DATA"]["MFN_TRAIN"]                   # MFN data/ train
MFN_VAL         = CONFIG["DATA"]["MFN_VAL"]                     # MFN data/ val
MFN_TEST        = CONFIG["DATA"]["MFN_TEST"]                    # MFN data/ test

YOLO_MODEL_PATH     = CONFIG["MODEL"]["YOLO_MODEL_PATH"]        #  đường dẫn tới yolo_best sau fine-tune
YOLO_CONF           = CONFIG["MODEL"]["YOLO_CONF"]              # độ tin cậy
YOLO_INPUT_SIZE     = CONFIG["IMAGE"]["YOLO_INPUT_SIZE"]        # kích thước trước khi detect

MIN_SIZE        = CONFIG["IMAGE"]["MIN_SIZE"]                   # kich thuoc toi thieu cua img
BRIGHTNESS      = CONFIG["IMAGE"]["BRIGHTNESS"]                 # nguong sang
IMG_EXTS        = CONFIG["IMAGE"]["IMG_EXTS"]                   # duoi file (.jpg)

USER_TRAIN  = CONFIG["IMAGE"]["USER_TRAIN"]             # Số đối tượng  train
USER_VAL    = CONFIG["IMAGE"]["USER_VAL"]               # Số đối tượng  val
USER_TEST   = CONFIG["IMAGE"]["USER_TEST"]              # Số đối tượng  test

TRAIN_RATIO = CONFIG["IMAGE"]["TRAIN_RATIO"]            # tỷ lệ train
VAL_RATIO   = CONFIG["IMAGE"]["VAL_RATIO"]              # tỷ lệ val
TEST_RATIO  = CONFIG["IMAGE"]["TEST_RATIO"]             # tỷ lệ test

# Số lượng ảnh tốt thiểu của từng người
MIN_IMAGES_PER_USER = CONFIG["IMAGE"]["MIN_IMAGES_PER_USER"]    # số ảnh tối thiểu mỗi user
MAX_IMAGES_PER_USER = CONFIG["IMAGE"]["MAX_IMAGES_PER_USER"]    # số ảnh tối đa mỗi user

# ================== Load YOLO model ==================
model = YOLO(os.path.join(YOLO_MODEL_PATH, "yolo_face_best.pt"))  # tai file yolo_best.pt sau khi fine-tune

# ================== 1.Check img ==================
def check_image_valid(path):
    """Kiểm tra ảnh hợp lệ: không hỏng, đủ sáng, đủ kích thước"""
    try:
        img = cv2.imread(path)
        if img is None:
            return None
        h, w = img.shape[:2]
        if h < MIN_SIZE or w < MIN_SIZE:
            return None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        if not (BRIGHTNESS[0] <= brightness <= BRIGHTNESS[1]):
            return None
        return img
    except:
        return None

# ================== 2.detect and crop img ==================
def detect_and_crop(img):
    """Detect face bằng YOLO, lọc ảnh có đúng 1 face, crop & resize"""
    results = model.predict(img, conf=YOLO_CONF, imgsz=YOLO_INPUT_SIZE, verbose=False)
    if results is None and len(results[0].boxes) == 0:
        return None           # nếu ko có khuôn mạt trong img loại bỏ
    boxes = results[0].boxes.xyxy.cpu().numpy()

    if len(boxes) != 1:   # chỉ giữ ảnh có đúng 1 face
        return None

    x1, y1, x2, y2 = boxes[0].astype(int)
    face = img[y1:y2, x1:x2]
    if face.size == 0:
        return None
    return face

# ================== 3.process img ==================
def process_user_dir(user_dir, save_dir, split=True):
    """Xử lý 1 thư mục con (một người)"""
    imgs = []
    for file in os.listdir(user_dir):
        if not any(file.lower().endswith(ext) for ext in IMG_EXTS):
            continue
        path = os.path.join(user_dir, file)
        img = check_image_valid(path)
        if img is None:
            continue
        face = detect_and_crop(img)
        if face is None:
            continue
        imgs.append(face)

    # bỏ qua nếu số ảnh < min
    if len(imgs) < MIN_IMAGES_PER_USER :
        print(f"[SKIP] {user_dir} has only {len(imgs)} valid images (< {MIN_IMAGES_PER_USER})")
        return  False
    # Kiểm ẩ va lây số lượng trong phạm vi
    if len(imgs) > MAX_IMAGES_PER_USER:
        imgs = imgs[:MAX_IMAGES_PER_USER]
    # shuffle ảnh
    np.random.shuffle(imgs)

    if split:  # tập train raw
        n_total = len(imgs)
        n_train = int(n_total * TRAIN_RATIO)
        # tao data cho train
        for idx, im in enumerate(imgs[:n_train]):
            save_path = os.path.join(save_dir["train"], os.path.basename(user_dir))
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, f"{idx}.jpg"), im)
        # tao data cho val
        for idx, im in enumerate(imgs[n_train:]):
            save_path = os.path.join(save_dir["val"], os.path.basename(user_dir))
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, f"{idx}.jpg"), im)

    else:  # tập test raw
        n_total = len(imgs)
        n_test  = int(n_total)
        for idx, im in enumerate(imgs[: n_test]):
            save_path = os.path.join(save_dir["test"], os.path.basename(user_dir))
            os.makedirs(save_path, exist_ok=True)
            cv2.imwrite(os.path.join(save_path, f"{idx}.jpg"), im)

    return True

def preprocess_mfn():
    # # Tạo data cho tập train/val
    # # kiểm tra truoc khi tao (train/val)
    # if len(os.listdir(MFN_TRAIN)) > 0:
    #     print(f" Đã có {MFN_TRAIN} yêu cau kiểm tra")
    #     return
    # if len(os.listdir(MFN_VAL)) > 0:
    #     print(f" Đã có {MFN_VAL} yêu cau kiểm tra")
    #     return
    # # train raw → train/val
    # count_users_train_val = 0
    # for user in tqdm(sorted(os.listdir(VGGFACE_TRAIN), reverse= True), desc="Processing VGGFACE_TRAIN"):
    #     if count_users_train_val >= USER_TRAIN:
    #         print("Thoát tạo train/val")
    #         break
    #     user_dir = os.path.join(VGGFACE_TRAIN, user)
    #     if os.path.isdir(user_dir):
    #         process_train_val = process_user_dir(user_dir, {"train": MFN_TRAIN, "val": MFN_VAL}, split=True)
    #         if process_train_val:
    #             count_users_train_val += 1
    # print("Đã tạo xong train/val")
    # tạo data cho tập test
    # kiem tra truoc khi tao (test)
    if len(os.listdir(MFN_TEST)) > 0:
        print(f" Đã có {MFN_TEST} yêu cau kiểm tra")
        return
    # test raw → test
    count_users_test = 0
    for user in tqdm(sorted(os.listdir(VGGFACE_TEST), reverse= True), desc="Processing VGGFACE_TEST"):
        if count_users_test >= USER_TEST:
            print("Thoát tạo test")
            break
        user_dir = os.path.join(VGGFACE_TEST, user)
        if os.path.isdir(user_dir):
            process_test = process_user_dir(user_dir, {"test": MFN_TEST}, split=False)
            if process_test:
                count_users_test += 1
    print("Đã tạo xong test")
if __name__ == "__main__":
    preprocess_mfn()

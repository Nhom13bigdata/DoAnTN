import os
from ultralytics import YOLO
from src.DATN.config.cau_hinh import CONFIG
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ----------------------------
# Cấu hình đường dẫn
# ----------------------------
WEIGHT_YOLO_DIR = CONFIG["MODEL"]["YOLO_MODEL_PATH"]                        # đờng dẫn tới thư mục Weight
DATA_YAML = CONFIG["TRAIN"]["YOLO_YAML"]                                    # đường dẫn tới yaml
BEST_WEIGHT_PATH = os.path.join(WEIGHT_YOLO_DIR, "yolo_face_best.pt")       # nơi lưu yolo tốt nhất
LOG_DIR = os.path.join(WEIGHT_YOLO_DIR, "log")                              # lưu log
RESULTS_DIR = os.path.join(LOG_DIR, "results.csv")                          # lưu kết quả sau mỗi epoch
BATCH_SIZE  = CONFIG["TRAIN"]["BATCH_SIZE"]                                 # Số ảnh train tại cùng thời điểm(batch size)
YOLO_SIZE   = CONFIG["IMAGE"]["YOLO_INPUT_SIZE"]                            # kích thước size trước khi train
EPOCHS      = CONFIG["TRAIN"]["EPOCHS"]                                     # Số vòng train ( trong conf epoch =  50)
os.makedirs(LOG_DIR, exist_ok=True)
LR0 = CONFIG["TRAIN"]["LR"]
MODEL_PATH  = CONFIG["MODEL"]["YOLO_FACE"]                                  # duong dan file yolov8m-face.pt
#----------------------------
# Load mô hình YOLOv8 pre-trained
model = YOLO(MODEL_PATH)
# ----------------------------
# Huấn luyện fine-tune
# ----------------------------
results = model.train(
    data=DATA_YAML,                             # đường dẫn tới yaml
    epochs=EPOCHS,                              # số vòng lặp(epoch)
    imgsz=YOLO_SIZE[0],                         # kích thước ảnh
    batch=BATCH_SIZE,                           # batch size
    project=WEIGHT_YOLO_DIR,                    # project & name: lưu checkpoint, log
    name="YOLO_finetune",
    exist_ok=True,
    device = CONFIG["TRAIN"]["DEVICE"],         # Kiểm tra có GPU thì dùng ko thì CPU
    save_period= 0,                             # không lưu checkpoint sau mỗi epoch
    patience= 15                                # số epoch để early stopping
)

# ----------------------------
# # Lưu weight tốt nhất
# # ----------------------------
# # weight tốt nhất (dựa vào mAP50 : độ chính xác trung bình)
# best_weight = os.path.join(WEIGHT_YOLO_DIR, "YOLO_finetune", "weights", "best.pt")
# if os.path.exists(best_weight):
#     if os.path.exists(BEST_WEIGHT_PATH):
#         os.remove(BEST_WEIGHT_PATH)
#     shutil.move(best_weight, BEST_WEIGHT_PATH)
#     print(f"Saved best weight to: {BEST_WEIGHT_PATH}")
# # results.csv lưu các các metrics của mỗi epoch
# results_path = os.path.join(WEIGHT_YOLO_DIR, "YOLO_finetune", "results.csv")
# if os.path.exists(results_path):
#     if os.path.exists(RESULTS_DIR):
#         os.remove(RESULTS_DIR)  # xóa file cũ nếu tồn tại
#     shutil.move(results_path, RESULTS_DIR)
#     print(f"Saved training results to: {RESULTS_DIR}")
# # ----------------------------
# # Vẽ biểu đồ loss train/val va biểu đồ đánh giá mô hình
# # ----------------------------
# metric_path = os.path.join(LOG_DIR, "results.csv")
# df = pd.read_csv(metric_path)
# epochs = df["epoch"]
#
# # Box Loss
# plt.figure(figsize=(8,5))
# plt.plot(epochs, df["train/box_loss"], 'b-o', label="Train Box Loss")
# plt.plot(epochs, df["val/box_loss"], 'r-o', label="Val Box Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Box Loss")
# plt.title("Box Loss - Train vs Validation")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(LOG_DIR, "box_loss.png"))  # lưu file
# plt.show()
#
# # DFL Loss
# plt.figure(figsize=(8,5))
# plt.plot(epochs, df["train/dfl_loss"], 'b-o', label="Train DFL Loss")
# plt.plot(epochs, df["val/dfl_loss"], 'r-o', label="Val DFL Loss")
# plt.xlabel("Epoch")
# plt.ylabel("DFL Loss")
# plt.title("DFL Loss - Train vs Validation")
# plt.legend()
# plt.grid(True)
# plt.savefig(os.path.join(LOG_DIR, "DFL_loss.png"))  # lưu file
# plt.show()
#
# # mAP@0.5
# plt.figure(figsize=(8,5))
# plt.plot(epochs, df["metrics/mAP50(B)"], 'c-o', label="mAP50")
# plt.xlabel("Epoch")
# plt.ylabel("mAP50")
# plt.title("metrics mAP50")
# plt.grid(True)
# plt.savefig(os.path.join(LOG_DIR, "mAP50.png"))  # lưu file
# plt.show()
#
# # mAP@0.5:0.95
# plt.figure(figsize=(8,5))
# plt.plot(epochs, df["metrics/mAP50-95(B)"], 'orange', label="mAP50-95")
# plt.xlabel("Epoch")
# plt.ylabel("mAP50-95")
# plt.title("metric mAP50-95")
# plt.grid(True)
# plt.savefig(os.path.join(LOG_DIR, "mAP50-95.png"))  # lưu file
# plt.show()
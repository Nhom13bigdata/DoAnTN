import os
from ultralytics import YOLO
import matplotlib.pyplot as plt
from src.DATN.config.cau_hinh import CONFIG

# ================== CẤU HÌNH ==================
# Thư mục và file
WEIGHT_YOLO_DIR = CONFIG["MODEL"]["YOLO_MODEL_PATH"]                        # đường dâ đến mục chứa wieght yolo
BEST_WEIGHT_PATH = os.path.join(WEIGHT_YOLO_DIR, "yolo_face_best.pt")       # best weight đã fine tune
TEST_IMG_DIR = CONFIG["DATA"]["IMGS_TEST"]                                  # img/test
TEST_LABEL_DIR = CONFIG["DATA"]["LABELS_TEST"]                              # img/label
YOLO_YAML = CONFIG["TRAIN"]["YOLO_YAML"]                                    # đường dẫn tới yaml test
# Tham số
BATCH_SIZE  = CONFIG["TRAIN"]["BATCH_SIZE"]                                  # Số img mỗi lần vào model
YOLO_SIZE   = CONFIG["IMAGE"]["YOLO_INPUT_SIZE"]                            # kích thuoc trước khi vao model
CONF_THRESH = CONFIG["MODEL"]["YOLO_CONF"]                                  # ngưỡng confidence dự đoán
LOG_DIR     = os.path.join(WEIGHT_YOLO_DIR, "log")                          # mục lưu các biểu đồ
os.makedirs(LOG_DIR, exist_ok=True)

# ================== LOAD MODEL ==================
model = YOLO(BEST_WEIGHT_PATH)
print(f"Loaded YOLOv8-Face model from: {BEST_WEIGHT_PATH}")

# ================== DỰ ĐOÁN & ĐÁNH GIÁ TEST ==================
results = model.val(
    data=YOLO_YAML,                     # đường dẫn tới yaml
    batch   = BATCH_SIZE,               # batch size
    imgsz   = YOLO_SIZE[0],             # kích thước imgz
    project = WEIGHT_YOLO_DIR,          # nơi lưu kết quả predict
    name    = "predict_yolo",           # ten noi luu ket qua du doán
    exist_ok= True,                     # ghi đè nếu mục đã có
    conf    = CONF_THRESH,              # ngưỡng độ tin cậy
    split   = "test",                   # thực hiên hiện đánh giá trên tập test
    save    = True,                     # lưu ảnh với bbox (.png)
)

# ================== ĐÁNH GIÁ MÔ HÌNH==================
#Lấy kết quả đánh giá
results_dict = results.results_dict
# kết quả đánh giá
recall, precision, mAP50, mAP5095 = results.mean_results()

print("=== Test set evaluation ===")
print(f"Precision      : {precision:.4f}")
print(f"Recall       : {recall:.4f}")
print(f"mAP@0.5      : {mAP50:.4f}")
print(f"mAP@0.5:0.95 : {mAP5095:.4f}")


# ================== VẼ BIỂU ĐỒ BAR CHART ==================
metrics_names =  ["Precision", "Recall","mAP@0.5", "mAP@0.5:0.95"]
metrics_values = [ precision, recall, mAP50, mAP5095]

plt.figure(figsize=(10,8))
bars = plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'magenta', 'cyan', 'orange'])
plt.ylim(0,1.1)
plt.title("YOLOv8-Face Test Set Metrics")
plt.ylabel("Value")
for bar, value in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1, f"{value:.2f}", ha='center', va='bottom')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Lưu biểu đồ
plot_path = os.path.join(LOG_DIR, "yolov8_face_test_metrics.png")
plt.savefig(plot_path)              # lưu kết quả của biểu đồ
plt.show()
print(f"Test metrics bar chart saved to: {plot_path}")

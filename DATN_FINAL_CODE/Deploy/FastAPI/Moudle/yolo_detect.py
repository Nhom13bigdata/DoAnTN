# ======================================================================
# FILE:         yolo_detect.py
# MÔ TẢ:        Module YOLOv8 để detect khuôn mặt trong ảnh
# ======================================================================
import cv2
import torch
import os
from ultralytics import YOLO
from src.DATN.config.cau_hinh import CONFIG
#  =====================CONFIG=====================
YOLO_MODEL_PATH = CONFIG["MODEL"]["YOLO_MODEL_PATH"]  # file yolo.pt

YOLO_CONF   =  0.8         # nguong conf
yolo_weight = os.path.join(YOLO_MODEL_PATH, "yolo_face_best.pt") # file trọng so


class YOLOFaceDetector:
    def __init__(self, model_path=yolo_weight, conf_threshold=YOLO_CONF):
        """
        model_path: đường dẫn model yolov8m-face
        conf_threshold: ngưỡng confidence để chọn khuôn mặt
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def detect_best_face(self, frames):
        """
        frames: list các ảnh np.array (OpenCV)
        return: cropped face image (np.array) nếu có, else -1
        """
        best_face = None
        best_conf = 0

        for frame in frames:
            # predict
            results = self.model.predict(frame, verbose=False)
            res = results[0]  # Results object

            if len(res.boxes) == 0:
                continue

            # lấy bbox và conf
            boxes = res.boxes.xyxy.cpu().numpy()  # (N,4)
            confs = res.boxes.conf.cpu().numpy()  # (N,)

            for (x1, y1, x2, y2), conf in zip(boxes, confs):
                if conf > self.conf_threshold and conf > best_conf: # conf_threshold ngưỡng tối thiểu (0.8), best_conf ( max)
                    best_conf = conf
                    # crop và convert sang int
                    best_face = frame[int(y1):int(y2), int(x1):int(x2)]

        if best_face is not None:
            return best_face
        else:
            return -1


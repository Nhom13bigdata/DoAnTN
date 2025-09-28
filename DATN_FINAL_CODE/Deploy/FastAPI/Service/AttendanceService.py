import base64
import cv2
import numpy as np
import sqlite3
import os
import threading
import queue
from datetime import datetime
from Deploy.FastAPI.Moudle.yolo_detect import YOLOFaceDetector
from Deploy.FastAPI.Moudle.MFN_embedding import MFNRecognizer
from src.DATN.config.cau_hinh import CONFIG

# --- DB paths ---
USERS_DB = CONFIG["DATA"]["DB_PATH"]
LOG_DB = os.path.join(CONFIG["DATA"]["EMBEDDING_DIR"], "log.db")

# --- Khởi tạo log.db nếu chưa có ---
if not os.path.exists(LOG_DB):
    conn = sqlite3.connect(LOG_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            time TEXT NOT NULL,
            status TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

# --- Module ---
yolo_detector = YOLOFaceDetector(conf_threshold=0.8)
mfn_recognizer = MFNRecognizer()

class AttendanceService:
    def __init__(self):
        self.running = False
        self.frame_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.processed_ids = set()  # chỉ dùng cho người quen
        self.yolo_detector = yolo_detector
        self.mfn_recognizer = mfn_recognizer
        self.lock = threading.Lock()
        self.worker_thread = None

        # Giữ DB connection mở lâu
        self.log_conn = sqlite3.connect(LOG_DB, check_same_thread=False)
        self.user_conn = sqlite3.connect(USERS_DB, check_same_thread=False)

    # --- Start attendance ---
    def start_attendance(self):
        if self.running:
            return {"status": "error", "message": "Điểm danh đang chạy."}
        self.running = True
        self.frame_queue.queue.clear()
        self.result_queue.queue.clear()
        self.processed_ids.clear()
        self.worker_thread = threading.Thread(target=self._process_frames_loop, daemon=True)
        self.worker_thread.start()
        return {"status": "success", "message": "Điểm danh đã bật. Frontend sẽ gửi frame."}

    # --- Stop attendance ---
    def stop_attendance(self):
        if not self.running:
            return {"status": "error", "message": "Điểm danh chưa chạy."}
        self.running = False
        self.frame_queue.queue.clear()
        self.result_queue.queue.clear()
        self.processed_ids.clear()
        return {"status": "success", "message": "Điểm danh đã dừng."}

    # --- Nhận frame từ frontend ---
    def add_frame(self, img_b64: str):
        if not self.running:
            return {"status": "error", "message": "Điểm danh chưa chạy."}
        try:
            img_bytes = base64.b64decode(img_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except:
            return {"status": "pending", "message": "Frame lỗi, bỏ qua"}

        if frame is None:
            return {"status": "pending", "message": "Frame decode lỗi, bỏ qua"}

        # Đưa frame vào queue để thread xử lý
        self.frame_queue.put(frame)

        # Nếu đã có kết quả → trả luôn
        try:
            result = self.result_queue.get_nowait()
            return {"status": "success", "detected": [result]}
        except queue.Empty:
            return {"status": "pending", "message": "Chờ xử lý frame..."}

    # --- Thread xử lý frame ---
    def _process_frames_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                best_face = self.yolo_detector.detect_best_face([frame])
                if type(best_face) != int:
                    user_id = self.mfn_recognizer.recognize_face(best_face)
                    if user_id >= 0 and user_id in self.processed_ids:
                        continue  # bỏ qua user đã điểm danh
                    result = self.mark_attendance(user_id)
                    if user_id >= 0:
                        self.processed_ids.add(user_id)
                else:
                    result = {
                        "status": "pending",
                        "message": "Không tìm thấy face"
                    }

            except Exception as e:
                result = {"status": "error", "message": str(e)}

            # Đưa kết quả vào queue để frontend lấy
            self.result_queue.put(result)

    # --- Ghi log điểm danh ---
    def mark_attendance(self, user_id):
        cursor = self.log_conn.cursor()
        if user_id >= 0:
            cursor.execute(
                "SELECT * FROM log WHERE user_id=? AND date(time)=date('now')",
                (user_id,)
            )
            log = cursor.fetchone()
            if log:
                cursor.execute(
                    "INSERT INTO log(user_id, time, status) VALUES (?, datetime('now'), ?)",
                    (user_id, "Đã điểm danh lại hôm nay")
                )
            else:
                cursor.execute(
                    "INSERT INTO log(user_id, time, status) VALUES (?, datetime('now'), ?)",
                    (user_id, "Điểm danh thành công")
                )
            self.log_conn.commit()

            cursor_user = self.user_conn.cursor()
            cursor_user.execute(
                "SELECT user_id, name, age, email, phone FROM users WHERE user_id=?",
                (user_id,)
            )
            user = cursor_user.fetchone()
            if user:
                user_id, name, age, email, phone = user
                return {
                    "user_id": user_id,
                    "name": name,
                    "age": age,
                    "email": email,
                    "phone": phone,
                    "time": str(np.datetime64('now')),
                    "status": "Điểm danh thành công"
                }
        else:
            # Người lạ
            return {
                "user_id": -1,
                "time": str(np.datetime64('now')),
                "status": "Người lạ chưa đăng ký"
            }

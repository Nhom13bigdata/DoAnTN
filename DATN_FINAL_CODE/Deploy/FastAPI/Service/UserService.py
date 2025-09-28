# user_service.py
import sqlite3
import os
import cv2
import numpy as np
import faiss
from Deploy.FastAPI.Moudle.yolo_detect import YOLOFaceDetector
from Deploy.FastAPI.Moudle.MFN_embedding import MFNRecognizer
from src.DATN.config.cau_hinh import CONFIG

# --- Cấu hình đường dẫn ---
USERS_DB       = CONFIG["DATA"]["DB_PATH"]                  # đường dẫn đén DB chứa thông tin user
EMBEDDING_DIR  = CONFIG["DATA"]["EMBEDDING_DIR"]            # duong dan den thu muc chua (faiss, db, ...)
IMG_USER_DIR   = os.path.join(EMBEDDING_DIR, "IMG_USER_DB")   # thư mục chứa ảnh user
FAISS_PATH     = CONFIG["DATA"]["FAISS_INDEX_PATH"]         # đường dẫn tới faiss chứa <id_class><embedding>
BASE_DIR       = CONFIG["DATA"]["ROOT"]                    # tuyet doi den du an
# --- Khởi tạo mô hình ---
yolo_detector  = YOLOFaceDetector(conf_threshold=0.8)
mfn_recognizer = MFNRecognizer()

# --- Tạo thư mục nếu chưa có ---
os.makedirs(IMG_USER_DIR, exist_ok=True)

class UserService:
    def __init__(self, db_path=USERS_DB, img_user_dir=IMG_USER_DIR):
        self.db_path = db_path
        self.img_user_dir = img_user_dir
        os.makedirs(self.img_user_dir, exist_ok=True)

        # Kiểm tra FAISS, nếu chưa có tạo mới
        if not os.path.exists(FAISS_PATH):
            dim = mfn_recognizer.embedding_size
            index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
            faiss.write_index(index, FAISS_PATH)
        else:
            index = faiss.read_index(FAISS_PATH)
        self.faiss_index = index

    # --- Xem thông tin user ---
    def view_user(self, user_id=None):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if user_id:
                cursor.execute(
                    "SELECT user_id, name, age, email, phone, image_path FROM users WHERE user_id=?",
                    (user_id,)
                )
            else:
                cursor.execute(
                    "SELECT user_id, name, age, email, phone, image_path FROM users"
                )

            rows = cursor.fetchall()

            if not rows:
                return {"status": "empty", "message": "Không tìm thấy người dùng."}

            users = [
                {
                    "user_id": r[0],
                    "name": r[1],
                    "age": r[2],
                    "email": r[3],
                    "phone": r[4],
                    "image_path": r[5]
                } for r in rows
            ]

            return {"status": "success", "users": users}

        except sqlite3.Error as e:
            # Bắt lỗi SQLite
            return {"status": "error", "message": f"Lỗi cơ sở dữ liệu: {str(e)}"}

        except Exception as e:
            # Bắt lỗi khác ngoài SQLite
            return {"status": "error", "message": f"Lỗi không xác định: {str(e)}"}

        finally:
            try:
                conn.close()
            except:
                pass
    # --- Thêm user mới ---
    def add_user(self, fullname, age, email, phone, img):

        face = yolo_detector.detect_best_face([img])
        if type(face) == int and face == -1:
            return {"status": "error", "message": "Không phát hiện khuôn mặt trong ảnh."}

        # Lấy id mới
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(user_id) FROM users")
        last_id = cursor.fetchone()[0]
        user_id = (last_id + 1) if last_id is not None else 1
        conn.close()

        # Lấy embedding MFN
        embedding = mfn_recognizer.get_embedding(face)

        # Thêm vào FAISS với class_id = user_id
        self.faiss_index.add_with_ids(embedding, np.array([user_id], dtype=np.int64))
        faiss.write_index(self.faiss_index, FAISS_PATH)

        # Lưu ảnh vào thư mục img_user_dir
        save_path = os.path.join(self.img_user_dir, f"user_{user_id}.jpg")                      # duong dan tuyet doi
        save_img_path = os.path.join("DATN_FINAL_CODE", os.path.relpath(save_path, BASE_DIR))   # duong dan tuong doi
        if not os.path.exists(save_path):
            cv2.imwrite(save_path, img)

        # Lưu thông tin vào DB
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO users(user_id, name, age, email, phone, image_path) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, fullname, age, email, phone, save_img_path)
        )
        conn.commit()
        conn.close()

        return {"status": "success", "message": f"Đã thêm người dùng {fullname} (ID={user_id})"}

    # --- Sửa thông tin user ---
    def update_user(self, user_id, fullname=None, age=None, email=None, phone=None, image=None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM users WHERE user_id=?", (user_id,))
        user = cursor.fetchone()
        if not user:
            conn.close()
            return {"status": "error", "message": f"Không tìm thấy người dùng với ID={user_id}"}
        old_image_path = user[0]

        update_fields = []
        params = []

        if fullname:
            update_fields.append("name=?")
            params.append(fullname)
        if age:
            update_fields.append("age=?")
            params.append(age)
        if email:
            update_fields.append("email=?")
            params.append(email)
        if phone:
            update_fields.append("phone=?")
            params.append(phone)
        if image.size == 0: # image đã là ndarray
            # image đã là ndarray
            if image.size == 0:
                return {"status": "error", "message": "Ảnh trống"}
            face = yolo_detector.detect_best_face([image])
            if type(face) == int and face == -1:
                conn.close()
                return {"status": "error", "message": "Không phát hiện khuôn mặt trong ảnh mới."}

            if self.faiss_index is not None:
                try:
                    # Lấy danh sách ID đã add vào index
                    existing_ids = faiss.vector_to_array(self.faiss_index.id_map)

                    # Nếu user_id đã tồn tại thì xóa đi
                    if user_id in existing_ids:
                        self.faiss_index.remove_ids(np.array([user_id], dtype=np.int64))
                        print(f"[FAISS] Đã xóa embedding cũ của user_id={user_id}")
                    else:
                        print(f"[FAISS] user_id={user_id} chưa có trong index, không cần xóa")
                except Exception as e:
                    print(f"[FAISS Warning] Lỗi khi check/remove id={user_id}: {e}")

            # Thêm embedding mới
            embedding = mfn_recognizer.get_embedding(face)
            self.faiss_index.add_with_ids(embedding, np.array([user_id], dtype=np.int64))
            faiss.write_index(self.faiss_index, FAISS_PATH)

            # Lưu ảnh mới
            save_path = os.path.join(self.img_user_dir, f"user_{user_id}.jpg")
            cv2.imwrite(save_path, image)
            save_img_path = os.path.join("DATN_FINAL_CODE", os.path.relpath(save_path, BASE_DIR))  # tuong doi
            update_fields.append("image_path=?")
            params.append(save_img_path)

        if update_fields:
            query = f"UPDATE users SET {', '.join(update_fields)} WHERE user_id=?"
            params.append(user_id)
            cursor.execute(query, params)
            conn.commit()
        conn.close()
        return {"status": "success", "message": f"Đã cập nhật người dùng ID={user_id}"}

    # --- Xóa user ---
    def delete_user(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT image_path FROM users WHERE user_id=?", (user_id,))
        user = cursor.fetchone()
        if not user:
            conn.close()
            return {"status": "error", "message": f"Không tìm thấy người dùng với ID={user_id}"}

        # Xóa ảnh
        path_img = user[0]
        # loại bỏ tên project dư
        if path_img.startswith("DATN_FINAL_CODE/") or path_img.startswith("DATN_FINAL_CODE\\"):
            path_img = path_img[len("DATN_FINAL_CODE\\"):]  # hoặc xử lý \

        path_img_real = os.path.join(BASE_DIR, path_img)
        if os.path.exists(path_img_real ):
            os.remove(path_img_real)

        # Xóa embedding trong FAISS
        self.faiss_index.remove_ids(np.array([user_id], dtype=np.int64))
        faiss.write_index(self.faiss_index, FAISS_PATH)

        # Xóa DB
        cursor.execute("DELETE FROM users WHERE user_id=?", (user_id,))
        conn.commit()
        conn.close()

        return {"status": "success", "message": f"Đã xóa người dùng ID={user_id}"}

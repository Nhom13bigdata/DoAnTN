# ======================================================================
# FILE:         create_user_db.py
# MÔ TẢ:        Tạo bảng thông tin đối tượng trong SQLite
# ======================================================================

import os
import sqlite3
import random
from src.DATN.config.cau_hinh import CONFIG
import unicodedata
import shutil
# ================CONFIG=====================
MFN_TRAIN = CONFIG["DATA"]["MFN_TRAIN"]  # input
DB_PATH = CONFIG["DATA"]["DB_PATH"]  # nơi tạo thông tin đối tượng
MAP_ID = CONFIG["DATA"]["MAP_ID"]
SELECTED_IMG_DIR = os.path.join(CONFIG["DATA"]["EMBEDDING_DIR"], "IMG_USER_DB") # thư mục lưu ảnh mẫu
BASE_DIR= CONFIG["DATA"]["ROOT"]           # tuyet doi
# Tạo thư mục lưu ảnh nếu chưa tồn tại
os.makedirs(SELECTED_IMG_DIR, exist_ok=True)

# ===== Database =====
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Tạo bảng users, thêm cột image_path để lưu ảnh đại diện
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER,
    email TEXT,
    phone TEXT,
    image_path TEXT
)
""")
conn.commit()

# Ví dụ list họ, tên đệm, tên
last_names = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Đặng", "Bùi", "Vũ", "Đỗ", "Ngô"]
middle_names = ["Văn", "Thị", "Hữu", "Minh", "Anh", "Quốc", "Thanh", "Ngọc", "Gia", "Xuân"]
first_names = ["Nam", "Hà", "Trang", "Hùng", "Tuấn", "Lan", "Linh", "Phương", "Khánh", "Dũng"]

# ===== Đọc file map_id.txt =====
map_id = {}
with open(MAP_ID, "r", encoding="utf-8") as f:
    for line in f:
        user_id, folder = line.strip().split()
        map_id[int(user_id)] = folder

# --- Hàm loại bỏ dấu tiếng Việt ---
def remove_vietnamese_accents(s):
    nfkd_form = unicodedata.normalize('NFKD', s)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])
# ===== Duyệt theo map_id và thêm user =====
for user_id, person_folder_name in map_id.items():
    person_folder = os.path.join(MFN_TRAIN, person_folder_name)
    if not os.path.isdir(person_folder):
        continue

    # ---- Sinh ngẫu nhiên tên đầy đủ ----
    fullname = f"{random.choice(last_names)} {random.choice(middle_names)} {random.choice(first_names)}"
    email_name = remove_vietnamese_accents(fullname).replace(" ", "").lower()
    email = f"{email_name}@example.com"
    phone = "0" + "".join([str(random.randint(0, 9)) for _ in range(9)])
    age = random.randint(18, 70)

    # ---- Lấy danh sách ảnh ----
    images = [f for f in os.listdir(person_folder) if os.path.isfile(os.path.join(person_folder, f))]
    image_path = None
    if images:
        selected_image = random.choice(images)
        src_path = os.path.join(person_folder, selected_image)

        # ---- Tạo tên ảnh mới là tên thư mục con ----
        ext = os.path.splitext(selected_image)[1]  # giữ phần mở rộng
        new_image_name = person_folder_name + ext
        new_img_path = os.path.join(SELECTED_IMG_DIR, new_image_name) # dia chi luu anh sau khi tao

        # ---- Copy ảnh sang thư mục mới ----
        shutil.copy2(src_path, new_img_path)
        # luu duong dan tuon doi vao cot imge_path
        image_path = os.path.join("DATN_FINAL_CODE", os.path.relpath(new_img_path, BASE_DIR)) # lay duong dan tuong doi

    cursor.execute("""
    INSERT OR IGNORE INTO users (user_id, name, age, email, phone, image_path) 
    VALUES (?, ?, ?, ?, ?, ?)
    """, (user_id, fullname, age, email, phone, image_path))

# Commit 1 lần cuối cùng
conn.commit()
conn.close()
print(f"Database created at {DB_PATH} !")
print(f"Selected images saved at {SELECTED_IMG_DIR} !")
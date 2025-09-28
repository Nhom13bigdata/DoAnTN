# ======================================================================
# FILE:        MFN_embedding.py
# MÔ TẢ:       Module MobileFaceNet nhan khuôn mặt trong ảnh tra ve embedding
# ======================================================================

import cv2
import torch
from src.DATN.config.cau_hinh import CONFIG
from PIL import Image
import os
import torch.nn.functional as F
from src.DATN.model.network.network_MFN import MobileFaceNet
import faiss
# ========================CONFIG=====================
MFN_MODEL_PATH  = CONFIG["MODEL"]["MFN_MODEL_PATH"]
MFN_WEIGHT = os.path.join(MFN_MODEL_PATH, "mfn_best.pt")
EMBEDDING_SIZE = CONFIG["TRAIN"]["EMBEDDING_SIZE"] # kích thước embedding
DEVICE = CONFIG["TRAIN"]["DEVICE"]
FAISS_PATH  = CONFIG["DATA"]["FAISS_INDEX_PATH"]        # duong dan den faiss idx
# transfrom
TRANSFROM = CONFIG["TRAIN"]["TRANSFORM"]
DEVICE  =CONFIG["TRAIN"]["DEVICE"]

class MFNRecognizer:
    """
    Module nhận diện khuôn mặt sử dụng MobileFaceNet và FAISS.
    """

    def __init__(self, model_path=MFN_WEIGHT, faiss_index_path=FAISS_PATH):
        """
        Khởi tạo model và FAISS index.

        model_path: đường dẫn đến checkpoint MobileFaceNet (.pt)
        faiss_index_path: đường dẫn đến file FAISS index
        """
        # Load MobileFaceNet
        self.model = MobileFaceNet()
        self.embedding_size = 128
        self.model.load_state_dict(torch.load(MFN_WEIGHT, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()  # đặt model ở chế độ eval (không train)

        # Load FAISS index đã được build trước
        self.faiss_index = faiss.read_index(faiss_index_path)

        # Chuỗi transform để chuẩn hóa ảnh trước khi đưa vào model
        self.transform = TRANSFROM  #chuan hóa truoc khi dua vao model

    def get_embedding(self, face_img):
        """
        Lấy embedding của một khuôn mặt từ ảnh.

        face_img: ảnh khuôn mặt (OpenCV, BGR)
        return: numpy array embedding
        """
        # Chuyển ảnh từ BGR (OpenCV) sang RGB và PIL Image
        img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

        # Áp dụng transform
        tensor = self.transform(img).unsqueeze(0)  # thêm batch dimension

        # Dự đoán embedding
        with torch.no_grad():  # không tính gradient
            embedding_tensor = self.model(tensor)  # output là tensor
            embedding_tensor = F.normalize(embedding_tensor, p=2, dim=1)  # chuẩn hóa L2
            embedding = embedding_tensor.cpu().numpy().astype('float32') # chuyển sang numpy để dùng với FAISS
        return embedding

    def recognize_face(self, face_img, distance_threshold=0.8):
        """
        Nhận diện khuôn mặt bằng embedding và FAISS.

        face_img: ảnh crop khuôn mặt
        distance_threshold: khoảng cách tối đa để coi là khớp
        return:
            -1 nếu không khớp (khuôn mặt mới)
            class_id nếu khớp với FAISS index
        """
        # Lấy embedding
        embedding = self.get_embedding(face_img)
        k = min(100, self.faiss_index.ntotal)  # k lớn, tối đa số vector trong index
        D, I = self.faiss_index.search(embedding, k=k)

        # Lọc chỉ id >= 301
        valid = [(dist, idx) for dist, idx in zip(D[0], I[0]) if idx >= 301]

        if not valid:
            return -1  

        # Lấy 1 người gần nhất trong danh sách hợp lệ
        nearest_dist, nearest_id = min(valid, key=lambda x: x[0])


        # Nếu muốn còn xét khoảng cách threshold:
        if nearest_dist > distance_threshold:
            return -1

        return int(nearest_id)




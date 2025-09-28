# main.py
from fastapi import FastAPI, Query, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
import os
import shutil
from fastapi.middleware.cors import CORSMiddleware
from Deploy.FastAPI.Service.AttendanceService import AttendanceService
from Deploy.FastAPI.Service.LogService import LogService
from Deploy.FastAPI.Service.UserService import UserService
import cv2
import numpy as np


app = FastAPI()
attendance_service = AttendanceService()
log_service = LogService()
user_service = UserService()

# Cho phép Front-end gọi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================== Điểm danh ========================
# --- Start attendance ---
@app.post("/attendance/start")
def start_attendance_api():
    res = attendance_service.start_attendance()
    # if res["status"] == "error":
    #     return JSONResponse(content=res, status_code=400)
    return res

# --- Stop attendance ---
@app.post("/attendance/stop")
def stop_attendance_api():
    res = attendance_service.stop_attendance()
    # if res["status"] == "error":
    #     return JSONResponse(content=res, status_code=400)
    return res

# --- Nhận frame từ frontend ---
@app.post("/attendance/frames")  # nho bo s
def receive_frame(frame: str = Body(..., embed=True)):
    """
    Frontend gửi frame base64.
    """
    res = attendance_service.add_frame(frame)
    return res
# ======================== Lịch sử logs ========================
@app.get("/logs/view")
def view_logs(
        year: int = Query(None, description="Năm muốn xem, ví dụ 2025"),
        month: int = Query(None, description="Tháng muốn xem, 1-12"),
        day: int = Query(None, description="Ngày muốn xem, 1-31")
):
    """
    Xem lịch sử điểm danh theo ngày/tháng/năm
    """
    result = log_service.view_history(year=year, month=month, day=day)
    return result


@app.delete("/logs/delete")
def delete_logs(
        year: int = Query(None, description="Năm muốn xóa, ví dụ 2025"),
        month: int = Query(None, description="Tháng muốn xóa, 1-12"),
        day: int = Query(None, description="Ngày muốn xóa, 1-31")
):
    """
    Xóa lịch sử điểm danh theo ngày/tháng/năm
    """
    result = log_service.delete_history(year=year, month=month, day=day)
    return result


# ======================== Quản lý user ========================
# --- Xem user ---
@app.get("/users/view")
def view_users(user_id: int = Query(None, description="ID người dùng muốn xem")):
    """
    Xem thông tin user theo ID hoặc tất cả
    """
    result = user_service.view_user(user_id=user_id)
    return result


# --- Thêm user ---
@app.post("/users/add")
def add_user(
        fullname: str = Form(...),
        age: int = Form(...),
        email: str = Form(...),
        phone: str = Form(...),
        image: UploadFile = File(...)
):
    """
    Thêm người dùng mới
    """
    # Đọc buffer ảnh trực tiếp
    file_bytes = np.frombuffer(image.file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return {"status": "error", "message": "Không đọc được ảnh."}
    else:
        result = user_service.add_user(fullname, age, email, phone, img)
        return result


# --- Sửa user ---
@app.put("/users/update")
def update_user(
        user_id: int = Form(...),
        fullname: str = Form(None),
        age: int = Form(None),
        email: str = Form(None),
        phone: str = Form(None),
        image: UploadFile = File(None)
):
    """
    Sửa thông tin user
    """

    if image:
        file_bytes = np.frombuffer(image.file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            return {"status": "error", "message": "Không đọc được ảnh."}

    result = user_service.update_user(
        user_id=user_id,
        fullname=fullname,
        age=age,
        email=email,
        phone=phone,
        image= img
    )
    return result


# --- Xóa user ---
@app.delete("/users/delete")
def delete_user(user_id: int = Query(..., description="ID người dùng muốn xóa")):
    """
    Xóa người dùng theo ID
    """
    result = user_service.delete_user(user_id)
    return result

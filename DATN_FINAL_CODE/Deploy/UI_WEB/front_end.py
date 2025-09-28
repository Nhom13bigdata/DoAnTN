import streamlit as st
import requests
import base64
import cv2
import numpy as np
from datetime import date, datetime
import time
import pandas as pd
import json
from io import BytesIO
import requests
import threading
from collections import deque
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit_autorefresh import st_autorefresh
import queue
# ===================== Cấu hình API =====================
BASE_URL = "http://localhost:8000"

# ===================== Cấu hình trang =====================
st.set_page_config(
    page_title="🎯 Hệ Thống Điểm Danh Tự Động",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== Custom CSS =====================
st.markdown("""
<style>
    /* Chỉnh tiêu đề chính */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem; /* Giảm padding để tiêu đề nhỏ hơn */
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 1rem; /* Giảm khoảng cách dưới */
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }

    /* Chỉnh kích thước font của tiêu đề chính */
    .main-header h1 {
        font-size: 1.8rem; /* Giảm kích thước font của H1 */
        margin: 0;
    }

    /* Chỉnh kích thước font của tiêu đề phụ */
    .main-header p {
        font-size: 0.9rem; /* Giảm kích thước font của thẻ P */
        margin: 0;
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    .success-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; /* Giảm khoảng cách giữa các tab */
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px; /* Giảm chiều cao của tab */
        padding: 0 15px; /* Giảm khoảng đệm ngang để tab hẹp hơn */
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .camera-frame {
        border: 3px solid #667eea;
        border-radius: 15px;
        padding: 10px;
        background: #f8f9fa;
    }
    .recording-indicator {
        background: #dc3545;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ===================== Header =====================
st.markdown("""
<div class="main-header">
    <h1>🎯 Hệ thống Điểm danh Tự Động</h1>
    <p>Điểm danh tự động bằng nhận diện khuôn mặt</p>
</div>
""", unsafe_allow_html=True)


# ===================== Helper Functions =====================
def get_status_color(status):
    colors = {
        'success': '#28a745',
        'warning': '#ffc107',
        'error': '#dc3545'
    }
    return colors.get(status, '#6c757d')


def get_status_text(status):
    texts = {
        'success': '✅ Thành công',
        'warning': '⚠️ Cảnh báo',
        'error': '❌ Lỗi'
    }
    return texts.get(status, '❓ Không xác định')


def make_api_call(endpoint, method="GET", data=None, files=None):
    """Helper function để gọi API"""
    try:
        url = f"{BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, params=data)
        elif method == "POST":
            response = requests.post(url, data=data, files=files)
        elif method == "PUT":
            response = requests.put(url, data=data, files=files)
        elif method == "DELETE":
            response = requests.delete(url, params=data)

        # Luôn trả về dict JSON
        return response.json()

    except Exception as e:
        return {"error": str(e)}

# ===================== Tabs =====================
tabs = st.tabs(["🎥 Điểm danh", "📜 Lịch sử Logs", "👤 Quản lý User"])

# ===================== TAB 1: Điểm danh =====================
if "cap" not in st.session_state:
    st.session_state.cap = None
if "camera_connected" not in st.session_state:
    st.session_state.camera_connected = False
if "recording" not in st.session_state:
    st.session_state.recording = False
if "running" not in st.session_state:
    st.session_state.running = False
if "processed_ids" not in st.session_state:
    st.session_state.processed_ids = set()
if "display_results" not in st.session_state:
    st.session_state.display_results = []

# Thread-safe queue thay cho result_queue trực tiếp
if "result_queue_threadsafe" not in st.session_state:
    st.session_state.result_queue_threadsafe = queue.Queue()

# ===================== Hàm gửi frame =====================
def send_frame(frame, running, processed_ids, result_queue):
    """Gửi frame nếu frontend đang chạy"""
    if not running:
        return

    try:
        # Resize + encode base64
        small_frame = cv2.resize(frame, (320, 320))
        _, buffer = cv2.imencode(".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        b64_frame = base64.b64encode(buffer).decode("utf-8")

        # Gửi frame lên backend
        res = requests.post(f"{BASE_URL}/attendance/frames",
                            json={"frame": b64_frame}, timeout=5).json()

        if res.get("status") == "success":
            for rec in res.get("detected", []):
                user_id = rec.get("user_id", f"unknown_{time.time()}")
                rec["name"] = rec.get("name") or "Người lạ"
                rec["time"] = rec.get("time") or datetime.now().strftime("%H:%M:%S")

                # Người quen đã điểm danh → bỏ qua
                if rec["name"] != "Người lạ" and user_id in processed_ids:
                    continue

                # Người lạ liên tiếp → bỏ qua
                if rec["name"] == "Người lạ":
                    # Kiểm tra queue thread-safe
                    if not result_queue.empty():
                        try:
                            last_rec = result_queue.queue[-1]
                            if last_rec["name"] == "Người lạ":
                                continue
                        except:
                            pass

                # Append vào queue thread-safe
                result_queue.put(rec)

                # Đánh dấu người quen đã điểm danh
                if rec["name"] != "Người lạ":
                    processed_ids.add(user_id)

    except Exception as e:
        print("Lỗi gửi frame:", e)

# ===================== Loop camera =====================
def live_camera_loop(cap, fps, recording, processed_ids, result_queue):
    frame_interval = 1.0 / fps
    while recording and cap and cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Gửi frame lên backend
        send_frame(frame.copy(), recording, processed_ids, result_queue)
        time.sleep(frame_interval)

# ===================== TAB 1: ĐIỂM DANH =====================
with tabs[0]:
    with st.container():
        st.header("🎥 Hệ thống Điểm danh")
        col_control, col_result = st.columns([1, 2])

        # ----- Cột điều khiển -----
        with col_control:
            st.subheader("📷 Điều khiển Camera")
            camera_options = {"0": "📱 Camera tích hợp", "1": "🔌 Camera USB", "2": "🌐 Camera IP"}
            selected_camera = st.selectbox("Chọn camera:", options=list(camera_options.keys()),
                                           format_func=lambda x: camera_options[x])

            fps = st.slider("🎚️ FPS (số frame/s gửi backend)", min_value=1, max_value=10, value=5)

            # Kết nối camera
            if st.button("🔌 Kết nối Camera"):
                try:
                    if st.session_state.cap:
                        st.session_state.cap.release()
                    cam_idx = int(selected_camera)
                    st.session_state.cap = cv2.VideoCapture(cam_idx)
                    st.session_state.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
                    st.session_state.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                    if st.session_state.cap.isOpened():
                        st.session_state.camera_connected = True
                        st.session_state.recording = True

                        # Thread live camera
                        threading.Thread(
                            target=live_camera_loop,
                            args=(
                                st.session_state.cap,
                                fps,
                                st.session_state.recording,
                                st.session_state.processed_ids,
                                st.session_state.result_queue_threadsafe
                            ),
                            daemon=True
                        ).start()
                        st.success(f"✅ Đã kết nối {camera_options[selected_camera]}")
                    else:
                        st.error("❌ Không mở được camera")
                        st.session_state.camera_connected = False
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối camera: {e}")
                    st.session_state.camera_connected = False

            # Start điểm danh
            if st.button("▶️ Bắt đầu", disabled=not st.session_state.camera_connected):
                try:
                    resp = requests.post(f"{BASE_URL}/attendance/start", timeout=5).json()
                    if resp.get("status") == "success":
                        st.session_state.running = True
                        st.session_state.processed_ids = set()
                        st.session_state.display_results = []
                        st.success("🎯 Điểm danh bắt đầu")
                    else:
                        st.error(f"❌ Backend trả lỗi: {resp.get('message','Unknown')}")
                except Exception as e:
                    st.error(f"❌ Lỗi kết nối backend: {e}")

            # Stop điểm danh
            if st.button("⏹️ Dừng"):
                st.session_state.recording = False
                st.session_state.running = False
                time.sleep(0.5)
                if st.session_state.cap:
                    st.session_state.cap.release()
                    st.session_state.cap = None
                st.session_state.camera_connected = False
                try:
                    requests.post(f"{BASE_URL}/attendance/stop", timeout=3)
                except:
                    pass
                st.success("⏹️ Điểm danh đã dừng")

        # ----- Cột kết quả -----
        with col_result:
            st.subheader("📊 Kết quả trực tiếp")
            # Refresh tự động mỗi 1s
            if st.session_state.get("running", False):
                st_autorefresh(interval=3000, key="attendance_refresh")

            # --- Lấy dữ liệu từ queue thread-safe ---
            results_to_display = []
            while not st.session_state.result_queue_threadsafe.empty():
                rec = st.session_state.result_queue_threadsafe.get()
                rec["user_id"]= rec.get("user_id", "N/A")
                rec["name"] = rec.get("name") or "Người lạ"
                rec["time"] = rec.get("time") or str(np.datetime64('now'))
                rec["email"] = rec.get("email", "N/A")
                rec["status"] = rec.get("status", "Không xác định")
                rec["is_unknown"] = rec["name"] == "Người lạ"

                results_to_display.append(rec)

                # Người quen -> đánh dấu processed
                if not rec["is_unknown"]:
                    st.session_state.processed_ids.add(rec.get("user_id"))

            # --- Hiển thị ---
            if not results_to_display:
                st.info("🔄 Chờ phát hiện khuôn mặt...")
            else:
                # Bắt đầu lưới
                html_content = '<div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap:15px;">'

                for rec in results_to_display:
                    is_unknown = rec["is_unknown"]
                    color = "#ff4d4d" if is_unknown else "#33cc33"  # đỏ nổi bật cho người lạ, xanh cho người quen
                    emoji = "😟" if is_unknown else "😃"

                    card_html = f"""
                        <div style="
                            background-color:{color};
                            border-radius:20px;
                            padding:20px;
                            color:#fff;
                            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
                            font-family: 'Arial', sans-serif;
                            transition: transform 0.2s;
                        " onmouseover="this.style.transform='scale(1.05)';" onmouseout="this.style.transform='scale(1)';">
                            <div style="font-size:60px; text-align:center; margin-bottom:10px;">{emoji}</div>
                            <div style="font-size:24px; font-weight:bold; text-align:center; margin-bottom:10px;">{rec.get('name', 'N/A')}</div>
                            <div style="font-size:16px; line-height:1.8;">
                                🕐 <b>Thời gian:</b> {rec.get('time', 'N/A')}<br>
                                🆔 <b>User ID:</b> {rec.get('user_id', 'N/A')}<br>
                                📧 <b>Email:</b> {rec.get('email', 'N/A')}<br>
                                🏷️ <b>Trạng thái:</b> {rec.get('status', 'Không xác định')}
                            </div>
                        </div>
                        """
                    html_content += card_html

                html_content += "</div>"

                st.markdown(html_content, unsafe_allow_html=True)

                # Xóa queue sau khi hiển thị
                st.session_state.display_results.clear()

# ===================== TAB 2: Lịch sử Logs =====================
with tabs[1]:
    st.header("📜 Quản lý Lịch sử Logs")

    if "current_logs" not in st.session_state:
        st.session_state.current_logs = []
    # Chọn ngày
    col1, col2 = st.columns([1, 2])
    with col1:
        selected_date = st.date_input("📅 Chọn ngày", value=date.today())

    with col2:
        col_btn1, col_btn2, col_btn3 = st.columns(3)

        # Tạo container để hiển thị bảng logs
        logs_container = st.container()

        # --- Nút xem logs ---
        with col_btn1:
            if st.button("🔍 Xem Logs", type="primary"):
                with st.spinner("Đang tải logs..."):
                    params = {
                        "year": selected_date.year,
                        "month": selected_date.month,
                        "day": selected_date.day
                    }
                    result = make_api_call("/logs/view", "GET", params)
                    if result.get("status") == "success":
                        st.session_state.current_logs = result['history']
                        st.success(f"✅ Tìm thấy {len(result['history'])} bản ghi")
                    else:
                        if result.get("status") == "empty":
                            st.warning(f"⚠️ {result['message']}")
                        else:
                            st.error(f"❌ {result['message']}")

        # --- Nút xóa logs ---
        with col_btn2:
            if st.button("🗑️ Xóa Logs"):
                if st.session_state.get("current_logs"):
                    params = {
                        "year": selected_date.year,
                        "month": selected_date.month,
                        "day": selected_date.day
                    }
                    result = make_api_call("/logs/delete", "DELETE", params)
                    st.session_state.current_logs = []
                    logs_container.empty()  # xóa ngay bảng + thống kê
                    if result.get("status") == "success":
                        st.success("✅ Đã xóa logs thành công!")
                    elif result.get("status") == "empty":
                        st.warning(f"⚠️ {result['message']}")
                    else:
                        st.error(f"❌ {result['message']}")
                else:
                    st.warning("⚠️ Không có logs để xóa")

        # --- Nút xuất CSV ---
        with col_btn3:
            if st.button("📥 Xuất CSV"):
                if st.session_state.get("current_logs"):
                    df = pd.DataFrame(st.session_state.current_logs)
                    csv_buffer = BytesIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    st.download_button(
                        label="💾 Tải xuống CSV",
                        data=csv_data,
                        file_name=f"attendance_logs_{selected_date.strftime('%Y-%m-%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("⚠️ Không có dữ liệu để xuất")

    # --- Hiển thị logs trong container ---
    with logs_container:
        if st.session_state.get("current_logs"):
            st.markdown("### 📊 Kết quả")
            df = pd.DataFrame(st.session_state.current_logs)
            st.dataframe(df, use_container_width=True)

            # Thống kê
            # col1, col2, col3, col4 = st.columns(4)
            # col1.metric("📊 Tổng bản ghi", len(df))
            # col2.metric("✅ Thành công", len(df[df['status'] == 'success']) if 'status' in df.columns else 0)
            # col3.metric("⚠️ Cảnh báo", len(df[df['status'] == 'empty']) if 'status' in df.columns else 0)
            # col4.metric("❌ Lỗi", len(df[df['status'] == 'error']) if 'status' in df.columns else 0)


# ===================== TAB 3: Quản lý User =====================
with tabs[2]:
    st.header("👤 Quản lý Người dùng")

    if "users_data" not in st.session_state:
        st.session_state.users_data = []

    # Bọc các thành phần bạn muốn làm ngắn vào một container riêng
    with st.container():
        st.markdown('<style>.compact-container { max-width: 200px; }</style>', unsafe_allow_html=True)
        st.markdown('<div class="compact-container">', unsafe_allow_html=True)
        action = st.selectbox(
            "🎯 Chọn hành động:",
            ["view-all", "view-user", "add-user", "edit-user", "delete-user"],
            format_func=lambda x: {
                "view-all": "👥 Xem tất cả",
                "view-user": "🔍 Xem người dùng",
                "add-user": "➕ Thêm người dùng",
                "edit-user": "✏️ Sửa người dùng",
                "delete-user": "🗑️ Xóa người dùng"
            }[x]
        )
        st.markdown('</div>', unsafe_allow_html=True)

    users_container = st.container()

    # -------------------- VIEW ALL --------------------
    if action == "view-all":
        if st.button("👥 Xem danh sách người dùng", type="primary"):
            res = make_api_call("/users/view", "GET")
            if res.get("status") == "success":
                st.session_state.users_data = res.get("users", [])
                st.success(f"Đã tải {len(st.session_state.users_data)} người dùng")
            else:
                st.session_state.users_data = []
                st.error("❌ Lỗi tải danh sách người dùng")
            users_container.empty()

    # -------------------- VIEW USER --------------------
    elif action == "view-user":
        user_id = st.text_input("🆔 Nhập ID người dùng:")
        if st.button("🔍 Tìm kiếm") and user_id:
            res = make_api_call("/users/view", "GET", {"user_id": user_id})
            if res.get("status") == "success" and res.get("users"):
                st.session_state.users_data = res.get("users", [])
                st.success(f"Đã tìm thấy người dùng ID {user_id}")
            else:
                st.session_state.users_data = []
                st.error(f"❌ Không tìm thấy người dùng ID {user_id}")
            users_container.empty()
    # -------------------- ADD USER --------------------
    elif action == "add-user":
        with st.form("add_user_form"):
            fullname = st.text_input("👤 Họ và tên *")
            age = st.number_input("🎂 Tuổi", min_value=1, max_value=120, value=25)
            email = st.text_input("📧 Email *")
            phone = st.text_input("📱 Số điện thoại")
            avatar = st.file_uploader("📷 Ảnh đại diện", type=["jpg", "png", "jpeg"])
            submitted = st.form_submit_button("➕ Thêm người dùng")
            if submitted and fullname and email:
                data = {"fullname": fullname, "age": age, "email": email, "phone": phone}
                files = {"image": (avatar.name, avatar, avatar.type)} if avatar else None
                res = make_api_call("/users/add", "POST", data, files)
                if res.get("status") == "success":
                    st.success("✅ Thêm thành công!")
                else:
                    st.error(res.get("message", "❌ Lỗi thêm người dùng"))
                    st.session_state.users_data = []

    # -------------------- EDIT USER --------------------
    elif action == "edit-user":
        with st.form("edit_user_form"):
            user_id = st.text_input("🆔 ID người dùng *")
            fullname = st.text_input("👤 Họ và tên")
            age = st.number_input("🎂 Tuổi", min_value=0, max_value=120, value=0)
            email = st.text_input("📧 Email")
            phone = st.text_input("📱 SĐT")
            avatar = st.file_uploader("📷 Ảnh mới", type=["jpg", "png", "jpeg"])
            submitted = st.form_submit_button("✏️ Cập nhật")
            if submitted and user_id:
                data = {"user_id": user_id, "fullname": fullname, "age": age, "email": email, "phone": phone}
                files = {"image": (avatar.name, avatar, avatar.type)} if avatar else None
                res = make_api_call("/users/update", "PUT", data, files)
                if res.get("status") == "success":
                    st.success("✅ Cập nhật thành công!")
                else:
                    st.error(res.get("message", "❌ Lỗi cập nhật"))
                    st.session_state.users_data = []

    # -------------------- DELETE USER --------------------
    elif action == "delete-user":
        user_id = st.text_input("🆔 Nhập ID người dùng")
        confirm = st.checkbox("✅ Xác nhận xóa")
        if st.button("🗑️ Xóa") and user_id and confirm:
            res = make_api_call("/users/delete", "DELETE", {"user_id": user_id})
            if res.get("status") == "success":
                st.success("✅ Xóa thành công!")
            else:
                st.error(res.get("message", "❌ Lỗi xóa"))
                st.session_state.users_data = []
            users_container.empty()
    # -------------------- HIỂN THỊ BẢNG NGƯỜI DÙNG --------------------
    with users_container:
        if st.session_state.users_data:
            users = st.session_state.users_data
            # Nếu users_data là dict, ép sang list
            if isinstance(users, dict):
                users = [users]
            elif not isinstance(users, list):
                users= list(users)

            # Tạo DataFrame
            df_users = pd.DataFrame(users).drop(columns=["image_path"], errors='ignore')

            # Ép cột user_id sang str để tránh lỗi PyArrow
            if "user_id" in df_users.columns:
                df_users["user_id"] = df_users["user_id"].astype(str)

            # Ép tất cả cột object sang string để tránh lỗi Arrow với mixed-type
            for col in df_users.select_dtypes(include=['object']).columns:
                df_users[col] = df_users[col].astype(str)

            # Hiển thị bảng AgGrid
            gb = GridOptionsBuilder.from_dataframe(df_users)
            grid_options = gb.build()

            AgGrid(
                df_users,
                gridOptions=grid_options,
                height=350,
                update_mode=GridUpdateMode.NO_UPDATE,
                fit_columns_on_grid_load=True
            )
        else:
            # Bọc thanh info vào container để làm ngắn
            st.markdown('<div class="compact-container">', unsafe_allow_html=True)
            st.info("Chưa có dữ liệu để hiển thị")
            st.markdown('</div>', unsafe_allow_html=True)
# ===================== Footer =====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; margin-top: 2rem;">
    💡 <strong>Hệ thống Điểm danh Tự Động</strong> - Powered by Computer Vision & Machine Learning<br>
    🚀 Phiên bản 2.0 | 📧 support@attendance.com
</div>
""", unsafe_allow_html=True)
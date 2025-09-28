"""
🎯 Attendance AI System - Local Launcher Script
Khởi chạy Backend API và Streamlit Frontend trên localhost
"""

import subprocess
import sys
import time
from pathlib import Path
import psutil

# ===================== Cấu hình =====================
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
FRONTEND_FILE = ROOT_DIR / "Deploy" / "UI_WEB" / "front_end.py"
BACKEND_MODULE = "Deploy.FastAPI.API.main_API:app"
BACKEND_PORT = 8000
FRONTEND_PORT = 8501

# ===================== Banner =====================
def print_banner():
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                   🎯 Automatic attendance system             ║
║                    Hệ thống Điểm danh Tự Động                ║
╠══════════════════════════════════════════════════════════════╣
║  🚀 Backend API: http://localhost:{BACKEND_PORT}             ║
║  📚 API Docs: http://localhost:{BACKEND_PORT}/docs           ║
║  🌐 Streamlit: http://localhost:{FRONTEND_PORT}              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

# ===================== Giải phóng cổng =====================
def free_port(port):
    """Kill process đang chiếm cổng"""
    for conn in psutil.net_connections():
        if conn.laddr.port == port:
            try:
                p = psutil.Process(conn.pid)
                print(f"⚠️ Killing process {conn.pid} đang chiếm cổng {port}")
                p.kill()
            except Exception as e:
                print(f"❌ Không thể kill PID {conn.pid}: {e}")

# ===================== Run Backend =====================
def run_backend():
    print("🚀 Starting Backend API...")
    proc = subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        BACKEND_MODULE,
        "--host", "127.0.0.1",   # chỉ localhost
        "--port", str(BACKEND_PORT),
        "--reload"
    ], cwd=str(ROOT_DIR))
    return proc

# ===================== Run Frontend =====================
def run_frontend():
    print("🌐 Starting Streamlit Frontend...")
    time.sleep(3)  # đợi backend khởi động
    proc = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        str(FRONTEND_FILE),
        "--server.port", str(FRONTEND_PORT),
        "--server.address", "localhost"  # chỉ localhost
    ])
    return proc

# ===================== Main =====================
def main():
    print_banner()

    # Giải phóng cổng nếu bị chiếm
    free_port(BACKEND_PORT)
    free_port(FRONTEND_PORT)

    print("\n🔄 Starting services...")

    # Chạy backend + frontend
    backend_proc = run_backend()
    frontend_proc = run_frontend()

    print("\n🎉 All services are running!")
    print(f"📱 Streamlit App: http://localhost:{FRONTEND_PORT}")
    print(f"📚 API Docs: http://localhost:{BACKEND_PORT}/docs")
    print("\n⚠️ Press Ctrl+C to stop all services")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all services...")
        if backend_proc:
            backend_proc.terminate()
            print("✅ Backend stopped")
        if frontend_proc:
            frontend_proc.terminate()
            print("✅ Frontend stopped")
        print("👋 Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    main()

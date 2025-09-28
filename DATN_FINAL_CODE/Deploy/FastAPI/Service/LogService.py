# log_service.py
import sqlite3
import os
from src.DATN.config.cau_hinh import CONFIG

LOG_DB = os.path.join(CONFIG["DATA"]["EMBEDDING_DIR"], "log.db")
USERS_DB = CONFIG["DATA"]["DB_PATH"]

class LogService:
    def __init__(self, log_db=LOG_DB, users_db=USERS_DB):
        self.log_db = log_db
        self.users_db = users_db

    def view_history(self, year=None, month=None, day=None):
        """Xem lịch sử log theo ngày/tháng/năm, kèm thông tin user"""
        # Kết nối log_db
        conn = sqlite3.connect(self.log_db)
        cursor = conn.cursor()
        import os
        if not os.path.exists(self.users_db):
            return {"status": "error", "message": "File users.db không tồn tại"}
        # Attach users.db
        cursor.execute(f"ATTACH DATABASE '{self.users_db}' AS users_db")

        # Query log + join với users_db.users để lấy thông tin chi tiết
        query = """
            SELECT l.user_id, us.name, us.age, us.email, us.phone, l.time, l.status
            FROM log l
            LEFT JOIN users_db.users us ON l.user_id = us.user_id 
            WHERE 1=1
        """
        params = []

        if year:
            query += " AND strftime('%Y', l.time)=?"
            params.append(str(year))
        if month:
            query += " AND strftime('%m', l.time)=?"
            params.append(f"{int(month):02d}")
        if day:
            query += " AND strftime('%d', l.time)=?"
            params.append(f"{int(day):02d}")

        query += " ORDER BY l.time DESC"
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {"status": "empty", "message": "Không tìm thấy dữ liệu lịch sử."}

        history = []
        for user_id, name, age, email, phone, time_str, status in rows:
            history.append({
                "user_id": user_id,
                "name": name if name else "Unknown",
                "age": age if age is not None else "Unknown",
                "email": email if email else "Unknown",
                "phone": phone if phone else "Unknown",
                "time": time_str,
                "status": status
            })

        return {"status": "success", "history": history}

    def delete_history(self, year=None, month=None, day=None):
        """Xóa lịch sử log theo ngày/tháng/năm"""
        try:
            conn = sqlite3.connect(self.log_db)
            cursor = conn.cursor()

            query = "DELETE FROM log WHERE 1=1"
            params = []

            if year:
                query += " AND strftime('%Y', time)=?"
                params.append(str(year))
            if month:
                query += " AND strftime('%m', time)=?"
                params.append(f"{int(month):02d}")
            if day:
                query += " AND strftime('%d', time)=?"
                params.append(f"{int(day):02d}")

            cursor.execute(query, params)
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()

            if deleted_count == 0:
                return {
                    "status": "empty",
                    "message": "Không có dữ liệu để xóa theo điều kiện này."
                }
            else:
                return {
                    "status": "success",
                    "message": f"Đã xóa {deleted_count} bản ghi theo yêu cầu."
                }

        except Exception as e:
            return {
                "status": "error",
                "message": f"Lỗi khi xóa log: {str(e)}"
            }

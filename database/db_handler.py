import sqlite3
import logging
from config import DB_NAME

logger = logging.getLogger(__name__)


class DBHandler:
    def __init__(self) -> None:
        self.db_path = DB_NAME
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Tạo kết nối SQLite với row_factory."""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn


    def _init_db(self) -> None:
        sql = """
            CREATE TABLE IF NOT EXISTS parking_log (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                plate        TEXT    NOT NULL,
                detect_time  TEXT    NOT NULL,
                event_type   TEXT    NOT NULL
            )
        """
        try:
            with self._get_connection() as conn:
                conn.execute(sql)
                conn.commit()
            logger.info("Database initialized: %s", self.db_path)
        except sqlite3.Error as exc:
            logger.error("SQLite init error: %s", exc)
            raise


    def insert_log(self, plate: str, event_type: str, detect_time: str) -> bool:
        sql = "INSERT INTO parking_log (plate, detect_time, event_type) VALUES (?, ?, ?)"
        try:
            with self._get_connection() as conn:
                conn.execute(sql, (plate, detect_time, event_type))
                conn.commit()
            logger.info("Logged → plate=%s  event=%s  time=%s", plate, event_type, detect_time)
            return True
        except sqlite3.Error as exc:
            logger.error("SQLite insert error: %s", exc)
            return False


    def get_all_logs(self) -> list[dict]:
        sql = """
            SELECT id, plate, detect_time, event_type
            FROM   parking_log
            ORDER  BY id DESC
        """
        try:
            with self._get_connection() as conn:
                rows = conn.execute(sql).fetchall()
            return [dict(r) for r in rows]
        except sqlite3.Error as exc:
            logger.error("SQLite get_all_logs error: %s", exc)
            return []

    def get_last_event(self, plate: str) -> str | None:
        sql = """
            SELECT event_type
            FROM   parking_log
            WHERE  plate = ?
            ORDER  BY id DESC
            LIMIT  1
        """
        try:
            with self._get_connection() as conn:
                row = conn.execute(sql, (plate,)).fetchone()
            return row["event_type"] if row else None
        except sqlite3.Error as exc:
            logger.error("SQLite get_last_event error: %s", exc)
            return None


    def get_stats(self) -> dict:
        try:
            with self._get_connection() as conn:
                total = conn.execute("SELECT COUNT(*) FROM parking_log").fetchone()[0]

                # Lấy event_type cuối cùng của mỗi biển số
                sql_last = """
                    SELECT event_type
                    FROM   parking_log
                    WHERE  id IN (
                        SELECT MAX(id)
                        FROM   parking_log
                        GROUP  BY plate
                    )
                """
                rows = conn.execute(sql_last).fetchall()
                in_lot = sum(1 for r in rows if r["event_type"] == "IN")

            return {"total": total, "in_lot": in_lot}
        except sqlite3.Error as exc:
            logger.error("SQLite get_stats error: %s", exc)
            return {"total": 0, "in_lot": 0}


    def clear_all_logs(self) -> bool:
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM parking_log")
                conn.execute("DELETE FROM sqlite_sequence WHERE name = 'parking_log'")
                conn.commit()
            logger.info("All logs cleared and ID auto-increment reset.")
            return True
        except sqlite3.Error as exc:
            logger.error("SQLite clear error: %s", exc)
            return False

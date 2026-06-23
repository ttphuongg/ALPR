import time
import threading
import logging
from tkinter import filedialog

import customtkinter as ctk

from gui.video_panel   import VideoPanel
from gui.info_panel    import InfoPanel
from gui.history_panel import HistoryPanel
import config

logger = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_BG        = "#0d1117"
_SURFACE   = "#161b22"
_SURFACE2  = "#21262d"
_ACCENT    = "#00d4aa"
_TEXT      = "#e6edf3"
_TEXT_MUTED= "#7d8590"


class ParkingApp(ctk.CTk):

    def __init__(self) -> None:
        super().__init__()

        # Cấu hình cửa sổ 
        self.title("Hệ Thống Quản Lý Bãi Đỗ Xe")
        self.geometry("1380x820")
        self.minsize(1100, 660)
        self.configure(fg_color=_BG)

        # Trạng thái nội bộ 
        self._current_frame      = None
        self._frame_lock         = threading.Lock()
        self._is_image_mode      = False   # True khi chọn ảnh tĩnh
        self._selected_path      = None    # Đường dẫn file đang chọn

        # Services (khởi tạo lazy)
        self._inference          = None   # PlateInference – load trong bg thread
        self._video_service      = None   # VideoService

        # Tạo widgets 
        self._build_header()
        self._build_body()

        # Khởi động 
        self.after(200, self._load_models_background)   # Load model sau 200ms
        self.after(33,  self._poll_video_frame)          # Poll frame 30fps

        # Đóng cửa sổ
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # Xây dựng giao diện 
    def _build_header(self) -> None:
        header = ctk.CTkFrame(
            self, fg_color=_SURFACE, corner_radius=0, height=58
        )
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        # Logo + tiêu đề
        left_box = ctk.CTkFrame(header, fg_color="transparent")
        left_box.pack(side="left", padx=18, pady=0)

        ctk.CTkLabel(
            left_box,
            text="HỆ THỐNG QUẢN LÝ BÃI ĐỖ XE",
            font=("Segoe UI", 17, "bold"),
            text_color=_ACCENT,
        ).pack(side="left")

        # Nhãn trạng thái hệ thống
        self._status_label = ctk.CTkLabel(
            header,
            text="⏳  Đang khởi tạo...",
            font=("Segoe UI", 11),
            text_color=_TEXT_MUTED,
        )
        self._status_label.pack(side="right", padx=18)

    def _build_body(self) -> None:
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=12, pady=10)

        # Cột trái: Video 
        left_col = ctk.CTkFrame(body, fg_color="transparent")
        left_col.pack(side="left", fill="both", expand=True)

        self.video_panel = VideoPanel(
            left_col,
            on_select_video=self._on_select_video,
            on_start=self._on_start,
            on_stop=self._on_stop,
        )
        self.video_panel.pack(fill="both", expand=True)

        # Cột phải: Info + History 
        right_col = ctk.CTkFrame(body, fg_color="transparent", width=490)
        right_col.pack(side="right", fill="both", padx=(12, 0))
        right_col.pack_propagate(False)

        self.info_panel = InfoPanel(right_col)
        self.info_panel.pack(fill="x")

        self.history_panel = HistoryPanel(
            right_col, on_clear=self._on_clear_history
        )
        self.history_panel.pack(fill="both", expand=True, pady=(10, 0))

    # Load models

    def _load_models_background(self) -> None:
        def _worker():
            from services.inference import PlateInference
            self._set_status("Đang tải models (lần đầu có thể mất vài giây)...")
            try:
                inference = PlateInference()
                self._inference = inference
                self.after(0, lambda: self._set_status(
                    "Sẵn sàng – Chọn video và nhấn [Bắt Đầu]"
                ))
                logger.info("PlateInference loaded successfully.")
            except Exception as exc:
                logger.error("Model load failed: %s", exc)
                self.after(0, lambda: self._set_status(
                    f"Lỗi tải model: {exc}"
                ))

        t = threading.Thread(target=_worker, name="ModelLoader", daemon=True)
        t.start()

    # Callbacks từ VideoService

    def _on_frame_received(self, frame) -> None:
        with self._frame_lock:
            self._current_frame = frame

    def _on_plate_detected(self, result: dict) -> None:
        just_logged = result.get("just_logged", False)

        # Cập nhật GUI trên main thread
        self.after(0, self.info_panel.update_plate, result)
        if just_logged:
            self.after(0, self._refresh_history)

    # Video frame polling (30fps)

    def _poll_video_frame(self) -> None:
        with self._frame_lock:
            frame = self._current_frame

        if frame is not None:
            self.video_panel.update_frame(frame)

        self.after(33, self._poll_video_frame)

    # Refresh dữ liệu 

    def _refresh_history(self) -> None:
        """Cập nhật bảng lịch sử và thẻ thống kê từ DB."""
        from database.db_handler import DBHandler
        # Dùng instance DB toàn cục (đã khởi tạo trong main.py)
        db = self._get_db()
        if db:
            logs  = db.get_all_logs()
            stats = db.get_stats()
            self.history_panel.update_table(logs)
            self.history_panel.update_stats(stats)

    # Nút điều khiển 

    def _on_select_video(self) -> None:
        IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        path = filedialog.askopenfilename(
            title="Chọn ảnh hoặc video",
            filetypes=[
                ("Video / ảnh",   "*.mp4 *.avi *.mov *.mkv *.wmv "
                                  "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
                ("Video files",   "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("Image files",   "*.jpg *.jpeg *.png *.bmp *.tiff *.tif *.webp"),
                ("All files",     "*.*"),
            ],
        )
        if not path:
            return

        # Dừng video cũ nếu đang chạy
        if self._video_service and self._video_service.is_running():
            self._on_stop()

        self._selected_path = path
        ext = "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""

        if ext in IMAGE_EXTS:
            # --- Chế độ ảnh tĩnh ---
            self._is_image_mode = True
            self.video_panel.set_image_path(path)
            filename = path.replace("\\", "/").split("/")[-1]
            self._set_status(f"Ảnh: {filename} – Đang nhận dạng...")
            logger.info("Image selected: %s", path)
            # Chạy inference trong background thread
            threading.Thread(
                target=self._process_image,
                args=(path,),
                name="ImageInference",
                daemon=True,
            ).start()
        else:
            # --- Chế độ video ---
            self._is_image_mode = False
            self.video_panel.set_video_path(path)
            filename = path.replace("\\", "/").split("/")[-1]
            self._set_status(f"Video: {filename} – Sẵn sàng")
            logger.info("Video selected: %s", path)

    def _process_image(self, path: str) -> None:
        """Chạy inference trên 1 ảnh tĩnh, gọi callback như video."""
        import cv2
        if self._inference is None:
            self.after(0, lambda: self._set_status(
                "Models chưa tải xong, vui lòng đợi rồi thử lại."
            ))
            return
        frame = cv2.imread(path)
        if frame is None:
            self.after(0, lambda: self._set_status("Ảnh không hợp lệ!"))
            return
        try:
            result = self._inference.process_frame(frame)
        except Exception as exc:
            logger.error("Image inference error: %s", exc)
            self.after(0, lambda: self._set_status(f"Lỗi nhận dạng: {exc}"))
            return

        filename = path.replace("\\", "/").split("/")[-1]
        if result and result.get("plate"):
            plate = result["plate"]
            from datetime import datetime
            detect_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Vẽ bbox lên ảnh
            annotated = self._draw_bbox_on_frame(frame, result)
            # Huần: gửi lên GUI
            self.after(0, lambda: self.video_panel.update_frame(annotated))
            from services.video_service import VideoService
            from database.db_handler import DBHandler
            db = self._get_db()
            event_type = "IN"
            if db:
                last_event = db.get_last_event(plate)
                event_type = "IN" if (last_event is None or last_event == "OUT") else "OUT"
                db.insert_log(plate, event_type, detect_time)
                self.after(0, self._refresh_history)
            payload = {
                "track_id":        0,
                "plate":           plate,
                "current_ocr":     plate,
                "bbox":            result.get("bbox"),
                "event_type":      event_type,
                "detect_time":     detect_time,
                "annotated_frame": annotated,
                "is_final":        True,
                "just_logged":     True,
            }
            self.after(0, self.info_panel.update_plate, payload)
            self.after(0, lambda: self._set_status(
                f"Ảnh: {filename} – Phát hiện: {plate} ({event_type})"
            ))
        else:
            self.after(0, lambda: self._set_status(
                f"Ảnh: {filename} – Không phát hiện biển số"
            ))

    @staticmethod
    def _draw_bbox_on_frame(frame, result: dict):
        """Vẽ bbox lên frame tạo ảnh annotated trả về."""
        import cv2
        annotated = frame.copy()
        bbox = result.get("bbox")
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (34, 197, 94), 2)
        return annotated

    def _on_start(self, video_path: str) -> None:
        if self._is_image_mode:
            self._set_status("Đây là chế độ ảnh – Nhấn [Chọn File] để chọn video.")
            return

        if not video_path:
            self._set_status("Vui lòng chọn video trước!")
            return

        if self._inference is None:
            self._set_status("Models chưa tải xong, vui lòng đợi...")
            return

        if self._video_service and self._video_service.is_running():
            self._set_status("Video đang phát...")
            return

        from services.video_service import VideoService
        db = self._get_db()
        if db is None:
            self._set_status("Lỗi kết nối database!")
            return

        self._video_service = VideoService(
            inference         = self._inference,
            db_handler        = db,
            on_frame_callback = self._on_frame_received,
            on_plate_callback = self._on_plate_detected,
        )
        self._video_service.set_video(video_path)
        self._video_service.start()
        self._set_status("Đang phát video và nhận diện biển số...")
        logger.info("VideoService started for: %s", video_path)

    def _on_stop(self) -> None:
        if self._video_service:
            self._video_service.stop()
            self._video_service = None

        with self._frame_lock:
            self._current_frame = None

        self.video_panel.show_placeholder()
        self._set_status("Đã dừng")
        logger.info("VideoService stopped.")

    def _on_clear_history(self) -> None:
        db = self._get_db()
        if db and db.clear_all_logs():
            self._refresh_history()
            self.info_panel.clear()
            self._set_status("Đã xóa lịch sử nhận diện")
            logger.info("All logs cleared by user.")

    def set_db(self, db) -> None:
        self._db = db
        # Tải dữ liệu cũ từ DB ngay khi nhận kết nối
        self._refresh_history()

    def _get_db(self):
        return getattr(self, "_db", None)

    def _set_status(self, msg: str) -> None:
        self._status_label.configure(text=msg)

    # Đóng ứng dụng 

    def _on_close(self) -> None:
        logger.info("Application closing...")
        self._on_stop()
        self.destroy()

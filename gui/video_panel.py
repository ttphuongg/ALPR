import cv2
import numpy as np
import customtkinter as ctk
from PIL import Image
from typing import Callable

_BG        = "#0d1117"
_SURFACE   = "#161b22"
_SURFACE2  = "#21262d"
_ACCENT    = "#00d4aa"
_BORDER    = "#30363d"
_TEXT      = "#e6edf3"
_TEXT_MUTED= "#7d8590"
_BTN_SEL   = "#1f6feb"
_BTN_SEL_H = "#388bfd"
_BTN_START = "#238636"
_BTN_START_H="#2ea043"
_BTN_STOP  = "#b91c1c"
_BTN_STOP_H= "#dc2626"


class VideoPanel(ctk.CTkFrame):

    def __init__(
        self,
        parent,
        on_select_video: Callable[[], None],
        on_start: Callable[[str], None],
        on_stop: Callable[[], None],
        **kwargs,
    ) -> None:
        super().__init__(parent, fg_color=_SURFACE, corner_radius=12, **kwargs)

        self._on_select_video = on_select_video
        self._on_start        = on_start
        self._on_stop         = on_stop
        self._video_path: str | None = None
        self._image_path: str | None = None  # Đường dẫn ảnh tĩnh
        self._photo           = None        # Giữ reference để tránh GC
        self._thumbnail_frame = None        # Frame thumbnail để hiển thị lại sau khi dừng
        self._video_started   = False

        self._build()

    def _build(self) -> None:
        # Thanh tiêu đề
        title_bar = ctk.CTkFrame(
            self, fg_color=_SURFACE2, corner_radius=0, height=42
        )
        title_bar.pack(fill="x")
        title_bar.pack_propagate(False)

        ctk.CTkLabel(
            title_bar,
            text="CAMERA / VIDEO FEED",
            font=("Segoe UI", 12, "bold"),
            text_color=_TEXT_MUTED,
        ).pack(side="left", padx=14, pady=10)

        # Thanh điều khiển 
        ctrl_bar = ctk.CTkFrame(
            self, fg_color=_SURFACE2, corner_radius=0, height=58
        )
        ctrl_bar.pack(fill="x", side="bottom")
        ctrl_bar.pack_propagate(False)

        # Vùng hiển thị video
        self._canvas_frame = ctk.CTkFrame(
            self, fg_color="#000000", corner_radius=0
        )
        self._canvas_frame.pack(fill="both", expand=True)

        self._video_label = ctk.CTkLabel(
            self._canvas_frame,
            text="",
            fg_color="#000000",
        )
        self._video_label.pack(fill="both", expand=True)

        # Placeholder khi chưa có video
        self._placeholder_frame = ctk.CTkFrame(
            self._canvas_frame, fg_color="transparent"
        )
        self._placeholder_frame.place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(
            self._placeholder_frame,
            text="📂",
            font=("Segoe UI", 52),
            text_color="#30363d",
        ).pack()
        ctk.CTkLabel(
            self._placeholder_frame,
            text="Chưa có video / ảnh",
            font=("Segoe UI", 16, "bold"),
            text_color="#30363d",
        ).pack(pady=(4, 2))
        ctk.CTkLabel(
            self._placeholder_frame,
            text="Nhấn [Chọn File] để bắt đầu",
            font=("Segoe UI", 11),
            text_color="#30363d",
        ).pack()

        self._btn_select = ctk.CTkButton(
            ctrl_bar,
            text="📂  Chọn File",
            font=("Segoe UI", 12, "bold"),
            fg_color=_BTN_SEL,
            hover_color=_BTN_SEL_H,
            width=145,
            height=38,
            corner_radius=8,
            command=self._on_select_video,
        )
        self._btn_select.pack(side="left", padx=(14, 6), pady=10)

        self._btn_start = ctk.CTkButton(
            ctrl_bar,
            text="▶   Bắt Đầu",
            font=("Segoe UI", 12, "bold"),
            fg_color=_BTN_START,
            hover_color=_BTN_START_H,
            width=130,
            height=38,
            corner_radius=8,
            command=self._handle_start,
        )
        self._btn_start.pack(side="left", padx=6, pady=10)

        self._btn_stop = ctk.CTkButton(
            ctrl_bar,
            text="⏹   Dừng",
            font=("Segoe UI", 12, "bold"),
            fg_color=_BTN_STOP,
            hover_color=_BTN_STOP_H,
            width=115,
            height=38,
            corner_radius=8,
            command=self._on_stop,
        )
        self._btn_stop.pack(side="left", padx=6, pady=10)

        # Tên file video đang chọn
        self._path_label = ctk.CTkLabel(
            ctrl_bar,
            text="Chưa chọn video",
            font=("Segoe UI", 10),
            text_color=_TEXT_MUTED,
        )
        self._path_label.pack(side="left", padx=12)

    # API 

    def set_video_path(self, path: str) -> None:
        self._video_path = path
        self._image_path = None
        filename = path.replace("\\", "/").split("/")[-1]
        self._path_label.configure(text=f"🎬  {filename}")
        self._video_started = False

        if path:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    self._thumbnail_frame = frame.copy()  # Lưu thumbnail
                    self.update_frame(frame)
                    self._video_started = False
                cap.release()

    def set_image_path(self, path: str) -> None:
        """Hiển thị một ảnh tĩnh lên panel."""
        self._image_path = path
        self._video_path = None
        filename = path.replace("\\", "/").split("/")[-1]
        self._path_label.configure(text=f"🖼️  {filename}")
        self._video_started = False

        frame = cv2.imread(path)
        if frame is not None:
            self._thumbnail_frame = frame.copy()  # Lưu thumbnail
            self.update_frame(frame)
            self._video_started = False

    def update_frame(self, frame: np.ndarray) -> None:
        try:
            if not self._video_started:
                self._placeholder_frame.place_forget()
                self._video_started = True

            # Kích thước hiển thị hiện tại
            w = self._video_label.winfo_width()
            h = self._video_label.winfo_height()
            if w < 10:
                w = 760
            if h < 10:
                h = 460

            # BGR → RGB → PIL
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # Giữ tỉ lệ khung hình (letterbox)
            img_w, img_h = pil_img.size
            ratio        = min(w / img_w, h / img_h)
            new_w        = max(1, int(img_w * ratio))
            new_h        = max(1, int(img_h * ratio))
            pil_img      = pil_img.resize((new_w, new_h), Image.LANCZOS)

            photo = ctk.CTkImage(
                light_image=pil_img,
                dark_image=pil_img,
                size=(new_w, new_h),
            )
            self._video_label.configure(image=photo, text="")
            self._photo = photo  # Giữ reference

        except Exception:
            pass

    def show_placeholder(self) -> None:
        """Sau khi dừng: hiển thị thumbnail file đang chọn (ảnh/frame đầu video),
        hoặc placeholder trống nếu chưa có file nào."""
        if self._thumbnail_frame is not None:
            self.update_frame(self._thumbnail_frame)
            self._video_started = False
            return

        self._video_label.configure(image=None, text="")
        self._photo = None
        self._video_started = False
        self._placeholder_frame.place(relx=0.5, rely=0.5, anchor="center")

    #Handler nội bộ 

    def _handle_start(self) -> None:
        if self._on_start:
            self._on_start(self._video_path or "")

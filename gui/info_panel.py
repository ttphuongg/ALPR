import customtkinter as ctk

_SURFACE   = "#161b22"
_SURFACE2  = "#21262d"
_ACCENT    = "#00d4aa"
_SUCCESS   = "#3fb950"
_DANGER    = "#f85149"
_BORDER    = "#30363d"
_TEXT      = "#e6edf3"
_TEXT_MUTED= "#7d8590"


class InfoPanel(ctk.CTkFrame):

    def __init__(self, parent, **kwargs) -> None:
        super().__init__(parent, fg_color=_SURFACE, corner_radius=12, **kwargs)
        self._build()


    def _build(self) -> None:
        # Tiêu đề
        header = ctk.CTkFrame(self, fg_color=_SURFACE2, corner_radius=0, height=42)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="THÔNG TIN NHẬN DIỆN",
            font=("Segoe UI", 12, "bold"),
            text_color=_TEXT_MUTED,
        ).pack(side="left", padx=14, pady=10)

        # Thẻ hiển thị biển số
        plate_card = ctk.CTkFrame(
            self, fg_color=_SURFACE2, corner_radius=10
        )
        plate_card.pack(fill="x", padx=14, pady=14)

        self._plate_header_label = ctk.CTkLabel(
            plate_card,
            text="BIỂN SỐ XE",
            font=("Segoe UI", 9, "bold"),
            text_color=_TEXT_MUTED,
        )
        self._plate_header_label.pack(pady=(12, 2))

        self._plate_label = ctk.CTkLabel(
            plate_card,
            text=" ",
            font=("Consolas", 32, "bold"),
            text_color=_ACCENT,
        )
        self._plate_label.pack(pady=(2, 14))

        # Hàng thông tin 2: Trạng thái + Thời gian
        info_row2 = ctk.CTkFrame(self, fg_color="transparent")
        info_row2.pack(fill="x", padx=14, pady=(0, 14))
        info_row2.columnconfigure(0, weight=1)
        info_row2.columnconfigure(1, weight=1)

        # Thẻ Trạng thái
        status_card = ctk.CTkFrame(info_row2, fg_color=_SURFACE2, corner_radius=10)
        status_card.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ctk.CTkLabel(
            status_card,
            text="TRẠNG THÁI",
            font=("Segoe UI", 9, "bold"),
            text_color=_TEXT_MUTED,
        ).pack(pady=(10, 2))
        self._status_label = ctk.CTkLabel(
            status_card,
            text="—",
            font=("Segoe UI", 20, "bold"),
            text_color=_TEXT_MUTED,
        )
        self._status_label.pack(pady=(2, 10))

        # Thẻ Thời gian
        time_card = ctk.CTkFrame(info_row2, fg_color=_SURFACE2, corner_radius=10)
        time_card.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ctk.CTkLabel(
            time_card,
            text="THỜI GIAN",
            font=("Segoe UI", 9, "bold"),
            text_color=_TEXT_MUTED,
        ).pack(pady=(10, 2))
        self._time_label = ctk.CTkLabel(
            time_card,
            text="—",
            font=("Segoe UI", 14, "bold"),
            text_color=_TEXT,
        )
        self._time_label.pack(pady=(2, 10))

    # API 

    def update_plate(self, result: dict) -> None:
        plate        = result.get("plate", "")
        current_ocr  = result.get("current_ocr", "")
        track_id     = result.get("track_id", "")
        event        = result.get("event_type", "")
        detect_time  = result.get("detect_time", "")
        is_final     = result.get("is_final", False)

        # Cập nhật tiêu đề và màu sắc biển số lớn
        if is_final:
            self._plate_header_label.configure(text="BIỂN SỐ XE (ĐÃ XÁC NHẬN)", text_color=_SUCCESS)
            self._plate_label.configure(text=plate if plate else " ", text_color=_ACCENT)
        else:
            self._plate_header_label.configure(text="BIỂN SỐ XE (ĐANG THEO DÕI)", text_color=_TEXT_MUTED)
            self._plate_label.configure(text=plate if plate else " ", text_color=_TEXT)

        # Trạng thái
        if event == "IN":
            self._status_label.configure(text="VÀO", text_color=_SUCCESS)
        elif event == "OUT":
            self._status_label.configure(text="RA", text_color=_DANGER)
        else:
            self._status_label.configure(text="—", text_color=_TEXT_MUTED)

        # Thời gian (chỉ hiển thị HH:MM:SS)
        if detect_time:
            time_str = detect_time.split(" ")[-1] if " " in detect_time else detect_time
            self._time_label.configure(text=time_str)
        else:
            self._time_label.configure(text="—")

    def clear(self) -> None:
        self._plate_header_label.configure(text="BIỂN SỐ XE", text_color=_TEXT_MUTED)
        self._plate_label.configure(text=" ", text_color=_ACCENT)
        self._status_label.configure(text="—", text_color=_TEXT_MUTED)
        self._time_label.configure(text="—")

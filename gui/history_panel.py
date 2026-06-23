import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as messagebox
import customtkinter as ctk
from typing import Callable

# Màu sắc 
_BG        = "#0d1117"
_SURFACE   = "#161b22"
_SURFACE2  = "#21262d"
_SURFACE3  = "#1a2133"
_ACCENT    = "#00d4aa"
_SUCCESS   = "#3fb950"
_DANGER    = "#f85149"
_BORDER    = "#30363d"
_TEXT      = "#e6edf3"
_TEXT_MUTED= "#7d8590"


class HistoryPanel(ctk.CTkFrame):
    def __init__(
        self,
        parent,
        on_clear: Callable[[], None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(parent, fg_color=_SURFACE, corner_radius=12, **kwargs)
        self._on_clear = on_clear
        self._build()


    def _build(self) -> None:
        # Tiêu đề 
        header = ctk.CTkFrame(self, fg_color=_SURFACE2, corner_radius=0, height=42)
        header.pack(fill="x")
        header.pack_propagate(False)

        ctk.CTkLabel(
            header,
            text="THỐNG KÊ & LỊCH SỬ",
            font=("Segoe UI", 12, "bold"),
            text_color=_TEXT_MUTED,
        ).pack(side="left", padx=14, pady=10)

        # Thẻ thống kê
        stats_row = ctk.CTkFrame(self, fg_color="transparent")
        stats_row.pack(fill="x", padx=14, pady=12)
        stats_row.columnconfigure(0, weight=1)
        stats_row.columnconfigure(1, weight=1)

        # Tổng lượt
        total_card = ctk.CTkFrame(stats_row, fg_color=_SURFACE2, corner_radius=10)
        total_card.grid(row=0, column=0, sticky="ew", padx=(0, 6))

        ctk.CTkLabel(
            total_card,
            text="TỔNG LƯỢT NHẬN DIỆN",
            font=("Segoe UI", 9, "bold"),
            text_color=_TEXT_MUTED,
        ).pack(pady=(10, 2))
        self._total_label = ctk.CTkLabel(
            total_card,
            text="0",
            font=("Segoe UI", 28, "bold"),
            text_color=_ACCENT,
        )
        self._total_label.pack(pady=(0, 10))

        # Xe trong bãi
        in_lot_card = ctk.CTkFrame(stats_row, fg_color=_SURFACE2, corner_radius=10)
        in_lot_card.grid(row=0, column=1, sticky="ew", padx=(6, 0))

        ctk.CTkLabel(
            in_lot_card,
            text="XE TRONG BÃI",
            font=("Segoe UI", 9, "bold"),
            text_color=_TEXT_MUTED,
        ).pack(pady=(10, 2))
        self._in_lot_label = ctk.CTkLabel(
            in_lot_card,
            text="0",
            font=("Segoe UI", 28, "bold"),
            text_color=_SUCCESS,
        )
        self._in_lot_label.pack(pady=(0, 10))

        # Thanh tiêu đề lịch sử + nút xóa
        list_header = ctk.CTkFrame(self, fg_color="transparent")
        list_header.pack(fill="x", padx=14, pady=(0, 6))

        ctk.CTkLabel(
            list_header,
            text="LỊCH SỬ NHẬN DIỆN",
            font=("Segoe UI", 11, "bold"),
            text_color=_TEXT_MUTED,
        ).pack(side="left")

        if self._on_clear:
            ctk.CTkButton(
                list_header,
                text="🗑  Xóa lịch sử",
                font=("Segoe UI", 10),
                fg_color="transparent",
                hover_color=_SURFACE2,
                border_width=1,
                border_color=_BORDER,
                text_color=_TEXT_MUTED,
                width=110,
                height=28,
                corner_radius=6,
                command=self._confirm_clear,
            ).pack(side="right")

        # Treeview 
        tree_container = ctk.CTkFrame(self, fg_color="transparent")
        tree_container.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self._setup_treeview_style()

        # Scrollbar
        scrollbar = ctk.CTkScrollbar(tree_container)
        scrollbar.pack(side="right", fill="y")

        self._tree = ttk.Treeview(
            tree_container,
            columns=("stt", "plate", "time", "event"),
            show="headings",
            style="Parking.Treeview",
            yscrollcommand=scrollbar.set,
        )
        scrollbar.configure(command=self._tree.yview)

        # Đầu cột
        self._tree.heading("stt",   text="#",         anchor="center")
        self._tree.heading("plate", text="Biển Số",   anchor="center")
        self._tree.heading("time",  text="Thời Gian", anchor="center")
        self._tree.heading("event", text="Loại",      anchor="center")

        # Độ rộng cột
        self._tree.column("stt",   width=40,  minwidth=30,  anchor="center", stretch=False)
        self._tree.column("plate", width=110, minwidth=90,  anchor="center", stretch=True)
        self._tree.column("time",  width=155, minwidth=130, anchor="center", stretch=True)
        self._tree.column("event", width=70,  minwidth=55,  anchor="center", stretch=False)

        # Tags màu sắc theo loại sự kiện
        self._tree.tag_configure("IN",     foreground=_SUCCESS)
        self._tree.tag_configure("OUT",    foreground=_DANGER)
        self._tree.tag_configure("IN_alt", foreground=_SUCCESS, background=_SURFACE3)
        self._tree.tag_configure("OUT_alt",foreground=_DANGER,  background=_SURFACE3)

        self._tree.pack(fill="both", expand=True)

    def _setup_treeview_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass  

        style.configure(
            "Parking.Treeview",
            background=_SURFACE2,
            foreground=_TEXT,
            fieldbackground=_SURFACE2,
            borderwidth=0,
            font=("Consolas", 11),
            rowheight=30,
        )
        style.configure(
            "Parking.Treeview.Heading",
            background=_BG,
            foreground=_ACCENT,
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            relief="flat",
            padding=6,
        )
        style.map(
            "Parking.Treeview",
            background=[("selected", "#2d3748")],
            foreground=[("selected", _TEXT)],
        )

    # API 

    def update_table(self, logs: list[dict]) -> None:
        # Xóa dữ liệu cũ
        for child in self._tree.get_children():
            self._tree.delete(child)

        # Thêm dữ liệu mới
        for idx, row in enumerate(logs):
            event    = row.get("event_type", "")
            is_alt   = (idx % 2 == 1)
            tag      = f"{event}_alt" if is_alt else event

            event_display = "VÀO" if event == "IN" else "RA"

            self._tree.insert(
                "",
                "end",
                iid=str(row.get("id", idx)),
                values=(
                    row.get("id", ""),
                    row.get("plate", ""),
                    row.get("detect_time", ""),
                    event_display,
                ),
                tags=(tag,),
            )

    def update_stats(self, stats: dict) -> None:
        self._total_label.configure(text=str(stats.get("total", 0)))
        self._in_lot_label.configure(text=str(stats.get("in_lot", 0)))

    # Xóa lịch sử 

    def _confirm_clear(self) -> None:
        if self._on_clear:
            self._on_clear()

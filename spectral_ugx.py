"""
Spectral-UGX: Fourier Currency Verifier
========================================
A Python desktop application for verifying Ugandan banknotes
using 2D FFT Power Spectral Density analysis.

Requirements:
    pip install customtkinter opencv-python numpy matplotlib pillow

Camera support uses OpenCV (cv2) for cross-platform compatibility.

Usage:
    python spectral_ugx.py
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk
import cv2
import numpy as np
import sqlite3
import json
import threading
import time
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ─── Theme ────────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# ─── Colour palette ──────────────────────────────────────────────────────────
C_BG       = "#060b14"
C_BG2      = "#0c1425"
C_BG3      = "#111d33"
C_GREEN    = "#00d28c"
C_CYAN     = "#00b8e6"
C_RED      = "#ff3a5c"
C_AMBER    = "#f5a623"
C_TEXT     = "#c8daf0"
C_DIMTEXT  = "#4a6080"
C_BORDER   = "#1a3050"

DB_PATH    = "ugx_currency.db"
DENOM_LIST = ["1,000", "2,000", "5,000", "10,000", "20,000", "50,000"]


# ══════════════════════════════════════════════════════════════════════════════
#  FFT / PSD UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def compute_psd(image_bgr: np.ndarray, bins: int = 128) -> np.ndarray:
    """
    Compute a 1-D radial Power Spectral Density from a BGR image.

    Pipeline
    --------
    1. Convert to greyscale.
    2. Apply 2-D FFT via numpy.
    3. Shift zero-frequency to centre.
    4. Take log-magnitude.
    5. Extract the mid-row (half-spectrum) as a 1-D PSD.

    Returns
    -------
    np.ndarray  shape (bins,)  float64
    """
    gray  = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    f     = np.fft.fft2(gray)
    fsh   = np.fft.fftshift(f)
    mag   = 20.0 * np.log(np.abs(fsh) + 1.0)

    h, w  = mag.shape
    psd   = mag[h // 2, w // 2:]          # half-row from DC
    return psd[:bins]


def pearson_correlation(a: np.ndarray, b: np.ndarray, n_bins: int = 50) -> float:
    """Pearson ρ on the first *n_bins* elements of two 1-D arrays."""
    n = min(len(a), len(b), n_bins)
    if n < 2:
        return 0.0
    a, b = a[:n], b[:n]
    corr_matrix = np.corrcoef(a, b)
    val = corr_matrix[0, 1]
    return float(val) if not np.isnan(val) else 0.0


def resize_for_display(frame_bgr: np.ndarray, max_w: int, max_h: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame_bgr


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════════

def db_init():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ugx_profiles (
            denomination TEXT PRIMARY KEY,
            psd_data     TEXT NOT NULL,
            threshold    REAL NOT NULL DEFAULT 0.85
        )
    """)
    conn.commit()
    conn.close()


def db_save_profile(denomination: str, psd: np.ndarray, threshold: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO ugx_profiles VALUES (?, ?, ?)",
        (denomination, json.dumps(psd.tolist()), threshold)
    )
    conn.commit()
    conn.close()


def db_load_profile(denomination: str):
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT psd_data, threshold FROM ugx_profiles WHERE denomination=?",
        (denomination,)
    ).fetchone()
    conn.close()
    if row:
        return np.array(json.loads(row[0])), row[1]
    return None, None


def db_all_denominations() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT denomination FROM ugx_profiles ORDER BY denomination").fetchall()
    conn.close()
    return [r[0] for r in rows]


def db_delete_profile(denomination: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM ugx_profiles WHERE denomination=?", (denomination,))
    conn.commit()
    conn.close()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class SpectralUGX(ctk.CTk):
    """Root window for the Spectral-UGX Currency Verifier."""

    def __init__(self):
        super().__init__()
        self.title("Spectral-UGX  │  Fourier Currency Verifier")
        self.geometry("1280x820")
        self.minsize(1000, 700)
        self.configure(fg_color=C_BG)

        db_init()

        # ── state ──────────────────────────────────────────────────────────
        self.current_image_bgr: np.ndarray | None = None   # full-res BGR
        self.current_psd:       np.ndarray | None = None
        self.camera_thread:     threading.Thread | None = None
        self.camera_running:    bool = False
        self.cap:               cv2.VideoCapture | None = None
        self.last_frame_bgr:    np.ndarray | None = None   # latest camera frame

        self._build_ui()
        self._refresh_profiles()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()

    # ─── SIDEBAR ──────────────────────────────────────────────────────────

    def _build_sidebar(self):
        sb = ctk.CTkFrame(self, width=240, corner_radius=0, fg_color=C_BG2,
                          border_width=1, border_color=C_BORDER)
        sb.grid(row=0, column=0, sticky="nsew")
        sb.grid_rowconfigure(20, weight=1)

        r = 0

        # Brand
        brand = ctk.CTkLabel(sb, text="SPECTRAL-UGX",
                             font=ctk.CTkFont("Courier", 18, "bold"),
                             text_color=C_GREEN)
        brand.grid(row=r, column=0, pady=(22, 2), padx=20, sticky="w"); r += 1

        ctk.CTkLabel(sb, text="FOURIER CURRENCY VERIFIER",
                     font=ctk.CTkFont(size=9), text_color=C_DIMTEXT
                     ).grid(row=r, column=0, padx=20, sticky="w"); r += 1

        self._sep(sb, r); r += 1

        # Denomination
        ctk.CTkLabel(sb, text="DENOMINATION", font=ctk.CTkFont("Courier", 10),
                     text_color=C_DIMTEXT).grid(row=r, column=0, padx=20, pady=(12, 2), sticky="w"); r += 1

        self.denom_var = ctk.StringVar(value="50,000")
        self.denom_menu = ctk.CTkOptionMenu(
            sb, values=DENOM_LIST, variable=self.denom_var,
            width=200, fg_color=C_BG3, button_color=C_BG3,
            button_hover_color="#1a2a45", text_color=C_TEXT,
            command=lambda _: self._on_denom_change()
        )
        self.denom_menu.grid(row=r, column=0, padx=20, pady=(0, 8), sticky="w"); r += 1

        self._sep(sb, r); r += 1

        # Threshold
        ctk.CTkLabel(sb, text="MIN. SIMILARITY THRESHOLD",
                     font=ctk.CTkFont("Courier", 10),
                     text_color=C_DIMTEXT).grid(row=r, column=0, padx=20, pady=(12, 2), sticky="w"); r += 1

        thresh_row = ctk.CTkFrame(sb, fg_color="transparent")
        thresh_row.grid(row=r, column=0, padx=20, sticky="ew"); r += 1

        self.thresh_var = tk.DoubleVar(value=85.0)
        self.thresh_label = ctk.CTkLabel(thresh_row, text="85%",
                                         font=ctk.CTkFont("Courier", 14, "bold"),
                                         text_color=C_GREEN, width=50)
        self.thresh_label.pack(side="right")

        self.thresh_slider = ctk.CTkSlider(
            thresh_row, from_=50, to=99, variable=self.thresh_var,
            width=140, progress_color=C_GREEN, button_color=C_GREEN,
            button_hover_color="#00ffb0",
            command=self._on_threshold_change
        )
        self.thresh_slider.pack(side="left", fill="x", expand=True)

        self._sep(sb, r); r += 1

        # Actions
        ctk.CTkLabel(sb, text="ACTIONS", font=ctk.CTkFont("Courier", 10),
                     text_color=C_DIMTEXT).grid(row=r, column=0, padx=20, pady=(12, 4), sticky="w"); r += 1

        self.save_btn = ctk.CTkButton(
            sb, text="★  Save as Golden Profile", width=200,
            fg_color=C_BG3, hover_color="#1a2a45",
            border_width=1, border_color=C_GREEN, text_color=C_GREEN,
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self.save_profile
        )
        self.save_btn.grid(row=r, column=0, padx=20, pady=3, sticky="w"); r += 1

        ctk.CTkButton(
            sb, text="⟳  Re-analyse Current", width=200,
            fg_color=C_BG3, hover_color="#1a2a45",
            border_width=1, border_color=C_CYAN, text_color=C_CYAN,
            font=ctk.CTkFont(size=13),
            command=self._analyse_and_plot
        ).grid(row=r, column=0, padx=20, pady=3, sticky="w"); r += 1

        ctk.CTkButton(
            sb, text="✕  Clear All Profiles", width=200,
            fg_color=C_BG3, hover_color="#1a2a45",
            border_width=1, border_color=C_RED, text_color=C_RED,
            font=ctk.CTkFont(size=13),
            command=self._clear_all_profiles
        ).grid(row=r, column=0, padx=20, pady=3, sticky="w"); r += 1

        self._sep(sb, r); r += 1

        # Stored profiles list
        ctk.CTkLabel(sb, text="STORED PROFILES", font=ctk.CTkFont("Courier", 10),
                     text_color=C_DIMTEXT).grid(row=r, column=0, padx=20, pady=(12, 4), sticky="w"); r += 1

        self.profiles_frame = ctk.CTkScrollableFrame(
            sb, width=200, height=130, fg_color=C_BG3,
            scrollbar_button_color=C_BG3, corner_radius=8
        )
        self.profiles_frame.grid(row=r, column=0, padx=20, pady=(0, 10), sticky="ew"); r += 1

        self.profiles_frame.grid_columnconfigure(0, weight=1)

        self._sep(sb, r); r += 1

        # Method note
        ctk.CTkLabel(
            sb,
            text="METHOD:\n2D FFT → log-magnitude\n→ 1D radial PSD\n→ Pearson ρ (0–50 bins)",
            font=ctk.CTkFont("Courier", 10), text_color=C_DIMTEXT,
            justify="left"
        ).grid(row=r, column=0, padx=20, pady=12, sticky="w")

    def _sep(self, parent, row):
        ctk.CTkFrame(parent, height=1, fg_color=C_BORDER
                     ).grid(row=row, column=0, padx=16, sticky="ew")

    # ─── MAIN AREA ────────────────────────────────────────────────────────

    def _build_main(self):
        main = ctk.CTkFrame(self, fg_color=C_BG, corner_radius=0)
        main.grid(row=0, column=1, padx=16, pady=16, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)

        # Verdict banner
        self._build_verdict_banner(main)

        # Input-mode tab bar
        self._build_tab_bar(main)

        # Content panels (image + chart)
        panels = ctk.CTkFrame(main, fg_color="transparent")
        panels.grid(row=2, column=0, sticky="nsew", pady=(0, 12))
        panels.grid_columnconfigure(0, weight=1)
        panels.grid_columnconfigure(1, weight=1)
        panels.grid_rowconfigure(0, weight=1)

        self._build_image_panel(panels)
        self._build_chart_panel(panels)

        # Frequency metrics row
        self._build_freq_metrics(main)

    # ─── VERDICT BANNER ───────────────────────────────────────────────────

    def _build_verdict_banner(self, parent):
        self.verdict_frame = ctk.CTkFrame(
            parent, fg_color=C_BG2, corner_radius=12,
            border_width=1, border_color=C_BORDER
        )
        self.verdict_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        self.verdict_frame.grid_columnconfigure(1, weight=1)

        self.verdict_icon = ctk.CTkLabel(
            self.verdict_frame, text="◌",
            font=ctk.CTkFont("Courier", 36, "bold"),
            text_color=C_DIMTEXT, width=60
        )
        self.verdict_icon.grid(row=0, column=0, padx=(20, 8), pady=16, rowspan=2)

        self.verdict_title = ctk.CTkLabel(
            self.verdict_frame, text="Awaiting Image Input",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=C_DIMTEXT, anchor="w"
        )
        self.verdict_title.grid(row=0, column=1, sticky="sw", padx=4, pady=(16, 0))

        self.verdict_sub = ctk.CTkLabel(
            self.verdict_frame,
            text="Upload a file or capture a photo to begin analysis",
            font=ctk.CTkFont("Courier", 11), text_color=C_DIMTEXT, anchor="w"
        )
        self.verdict_sub.grid(row=1, column=1, sticky="nw", padx=4, pady=(0, 16))

        self.verdict_score = ctk.CTkLabel(
            self.verdict_frame, text="—",
            font=ctk.CTkFont("Courier", 32, "bold"),
            text_color=C_DIMTEXT, width=90
        )
        self.verdict_score.grid(row=0, column=2, padx=(8, 4), pady=16, rowspan=2)

        ctk.CTkLabel(
            self.verdict_frame, text="SIMILARITY",
            font=ctk.CTkFont("Courier", 9), text_color=C_DIMTEXT
        ).grid(row=1, column=2, sticky="n", padx=(8, 20))

    # ─── TAB BAR ──────────────────────────────────────────────────────────

    def _build_tab_bar(self, parent):
        tab_frame = ctk.CTkFrame(parent, fg_color=C_BG3, corner_radius=10,
                                 border_width=1, border_color=C_BORDER)
        tab_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        self.active_tab = tk.StringVar(value="upload")

        self.tab_upload_btn = ctk.CTkButton(
            tab_frame, text="📁  Upload File",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=C_BG2, hover_color="#1a2a45",
            text_color=C_GREEN, border_width=1, border_color=C_GREEN,
            corner_radius=8, command=lambda: self._switch_tab("upload")
        )
        self.tab_upload_btn.pack(side="left", padx=8, pady=6, fill="x", expand=True)

        self.tab_camera_btn = ctk.CTkButton(
            tab_frame, text="📷  Camera / Phone",
            font=ctk.CTkFont(size=13),
            fg_color="transparent", hover_color="#1a2a45",
            text_color=C_DIMTEXT, corner_radius=8,
            command=lambda: self._switch_tab("camera")
        )
        self.tab_camera_btn.pack(side="left", padx=8, pady=6, fill="x", expand=True)

    def _switch_tab(self, mode: str):
        self.active_tab.set(mode)
        is_upload = mode == "upload"

        # Upload tab styling
        self.tab_upload_btn.configure(
            fg_color=C_BG2 if is_upload else "transparent",
            text_color=C_GREEN if is_upload else C_DIMTEXT,
            border_width=1 if is_upload else 0
        )
        # Camera tab styling
        self.tab_camera_btn.configure(
            fg_color=C_BG2 if not is_upload else "transparent",
            text_color=C_GREEN if not is_upload else C_DIMTEXT,
            border_width=1 if not is_upload else 0,
            border_color=C_GREEN if not is_upload else C_BORDER
        )

        if is_upload:
            self.camera_panel.pack_forget()
            self.upload_inner.pack(fill="both", expand=True)
            if self.camera_running:
                self._stop_camera()
        else:
            self.upload_inner.pack_forget()
            self.camera_panel.pack(fill="both", expand=True)

    # ─── IMAGE PANEL ──────────────────────────────────────────────────────

    def _build_image_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=C_BG2, corner_radius=12,
                             border_width=1, border_color=C_BORDER)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="◼  INPUT ROI",
                     font=ctk.CTkFont("Courier", 10), text_color=C_DIMTEXT
                     ).grid(row=0, column=0, padx=14, pady=(10, 4), sticky="w")

        content = ctk.CTkFrame(frame, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        content.pack_propagate(False)

        # ── UPLOAD inner ──────────────────────────────────────────────
        self.upload_inner = ctk.CTkFrame(content, fg_color="transparent")
        self.upload_inner.pack(fill="both", expand=True)

        self.drop_frame = ctk.CTkFrame(
            self.upload_inner, fg_color=C_BG3, corner_radius=8,
            border_width=1, border_color=C_BORDER
        )
        self.drop_frame.pack(fill="both", expand=True)

        ctk.CTkLabel(self.drop_frame, text="🪙",
                     font=ctk.CTkFont(size=36)).pack(pady=(30, 6))
        ctk.CTkLabel(self.drop_frame, text="Click to browse for an image",
                     font=ctk.CTkFont(size=13), text_color=C_TEXT).pack()
        ctk.CTkLabel(self.drop_frame, text="JPG · PNG · BMP · WEBP",
                     font=ctk.CTkFont("Courier", 10), text_color=C_DIMTEXT).pack(pady=4)

        ctk.CTkButton(
            self.drop_frame, text="Browse File",
            fg_color=C_GREEN, text_color=C_BG, hover_color="#00ffb0",
            font=ctk.CTkFont(size=13, weight="bold"),
            command=self._upload_file
        ).pack(pady=(8, 30))

        self.upload_preview_label = ctk.CTkLabel(self.upload_inner, text="", image=None)

        # ── CAMERA inner ──────────────────────────────────────────────
        self.camera_panel = ctk.CTkFrame(content, fg_color="transparent")

        # video feed label
        self.cam_feed_label = ctk.CTkLabel(
            self.camera_panel, text="Camera inactive",
            fg_color=C_BG3, corner_radius=8,
            font=ctk.CTkFont("Courier", 12), text_color=C_DIMTEXT,
            width=380, height=220
        )
        self.cam_feed_label.pack(fill="x", pady=(0, 6))

        cam_btns = ctk.CTkFrame(self.camera_panel, fg_color="transparent")
        cam_btns.pack(fill="x")

        self.start_cam_btn = ctk.CTkButton(
            cam_btns, text="▶ Start",
            fg_color=C_BG3, hover_color="#1a2a45",
            border_width=1, border_color=C_GREEN, text_color=C_GREEN,
            command=self._start_camera, width=90
        )
        self.start_cam_btn.pack(side="left", padx=(0, 4))

        self.snap_btn = ctk.CTkButton(
            cam_btns, text="📸 Capture",
            fg_color=C_BG3, hover_color="#1a2a45",
            border_width=1, border_color=C_CYAN, text_color=C_CYAN,
            command=self._snap_photo, state="disabled", width=110
        )
        self.snap_btn.pack(side="left", padx=4)

        self.stop_cam_btn = ctk.CTkButton(
            cam_btns, text="■ Stop",
            fg_color=C_BG3, hover_color="#1a2a45",
            border_width=1, border_color=C_RED, text_color=C_RED,
            command=self._stop_camera, state="disabled", width=90
        )
        self.stop_cam_btn.pack(side="left", padx=4)

        self.cam_status_label = ctk.CTkLabel(
            self.camera_panel, text="",
            font=ctk.CTkFont("Courier", 10), text_color=C_DIMTEXT
        )
        self.cam_status_label.pack(pady=4)

        self.cam_preview_label = ctk.CTkLabel(self.camera_panel, text="", image=None)

    # ─── CHART PANEL ──────────────────────────────────────────────────────

    def _build_chart_panel(self, parent):
        frame = ctk.CTkFrame(parent, fg_color=C_BG2, corner_radius=12,
                             border_width=1, border_color=C_BORDER)
        frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        frame.grid_rowconfigure(1, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(frame, text="◼  FOURIER FREQUENCY PEAKS",
                     font=ctk.CTkFont("Courier", 10), text_color=C_DIMTEXT
                     ).grid(row=0, column=0, padx=14, pady=(10, 4), sticky="w")

        self.fig = Figure(figsize=(5.5, 3.5), facecolor=C_BG2)
        self.ax  = self.fig.add_subplot(111)
        self._style_ax(self.ax)

        self.chart_canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.chart_canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew",
                                               padx=8, pady=(0, 10))

    def _style_ax(self, ax):
        ax.set_facecolor(C_BG2)
        ax.tick_params(colors=C_DIMTEXT, labelsize=8)
        ax.spines[:].set_color(C_BORDER)
        ax.grid(True, color=C_BORDER, linewidth=0.4, alpha=0.6)
        ax.set_xlabel("Frequency bin", color=C_DIMTEXT, fontsize=9)
        ax.set_ylabel("Log magnitude", color=C_DIMTEXT, fontsize=9)

    # ─── FREQUENCY METRICS ROW ────────────────────────────────────────────

    def _build_freq_metrics(self, parent):
        row_frame = ctk.CTkFrame(parent, fg_color="transparent")
        row_frame.grid(row=3, column=0, sticky="ew")
        row_frame.grid_columnconfigure((0, 1, 2), weight=1)

        specs = [
            ("LOW FREQ  (0–16)",   "f_low",  C_GREEN),
            ("MID FREQ  (17–32)",  "f_mid",  C_CYAN),
            ("HIGH FREQ (33–50)", "f_hi",   C_AMBER),
        ]
        self._freq_labels = {}
        for i, (label, key, color) in enumerate(specs):
            cell = ctk.CTkFrame(row_frame, fg_color=C_BG2, corner_radius=8,
                                border_width=1, border_color=C_BORDER)
            cell.grid(row=0, column=i, padx=4 if i else (0, 4), sticky="ew")
            ctk.CTkLabel(cell, text=label, font=ctk.CTkFont("Courier", 9),
                         text_color=C_DIMTEXT).pack(anchor="w", padx=12, pady=(8, 0))
            val_lbl = ctk.CTkLabel(cell, text="—",
                                   font=ctk.CTkFont("Courier", 16, "bold"),
                                   text_color=color)
            val_lbl.pack(anchor="w", padx=12, pady=(0, 8))
            self._freq_labels[key] = val_lbl

    # ══════════════════════════════════════════════════════════════════════
    #  UPLOAD FLOW
    # ══════════════════════════════════════════════════════════════════════

    def _upload_file(self):
        path = filedialog.askopenfilename(
            title="Select Banknote Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff"), ("All files", "*.*")]
        )
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Could not read the image file.")
            return
        self._ingest_image(img)
        self._show_upload_preview(img)

    def _show_upload_preview(self, img_bgr: np.ndarray):
        """Show the uploaded image inside the upload panel."""
        disp = resize_for_display(img_bgr, 380, 220)
        rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb)
        self._upload_pil_ref = ctk.CTkImage(pil, size=(pil.width, pil.height))

        self.drop_frame.pack_forget()
        self.upload_preview_label.configure(image=self._upload_pil_ref, text="")
        self.upload_preview_label.pack(fill="both", expand=True)

    # ══════════════════════════════════════════════════════════════════════
    #  CAMERA FLOW
    # ══════════════════════════════════════════════════════════════════════

    def _start_camera(self):
        if self.camera_running:
            return
        # Try cameras 0..3
        for idx in range(4):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                self.cap = cap
                break
        else:
            messagebox.showerror("Camera Error", "No camera found. Please plug in a webcam or allow access.")
            return

        self.camera_running = True
        self.start_cam_btn.configure(state="disabled")
        self.snap_btn.configure(state="normal")
        self.stop_cam_btn.configure(state="normal")
        self.cam_status_label.configure(text="● Camera live — position the banknote and press Capture",
                                        text_color=C_GREEN)

        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()

    def _camera_loop(self):
        """Background thread: read frames, push to UI via after()."""
        while self.camera_running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.last_frame_bgr = frame.copy()
                disp = resize_for_display(frame, 380, 220)
                rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
                pil  = Image.fromarray(rgb)
                ctk_img = ctk.CTkImage(pil, size=(pil.width, pil.height))
                self.after(0, self._update_cam_feed, ctk_img)
            time.sleep(1 / 30)

    def _update_cam_feed(self, ctk_img):
        self._cam_feed_ref = ctk_img
        self.cam_feed_label.configure(image=ctk_img, text="")

    def _snap_photo(self):
        if self.last_frame_bgr is None:
            messagebox.showwarning("Camera", "No frame captured yet — wait a moment.")
            return
        frame = self.last_frame_bgr.copy()
        self._ingest_image(frame)
        # Show snapshot preview below controls
        disp = resize_for_display(frame, 380, 140)
        rgb  = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        pil  = Image.fromarray(rgb)
        self._snap_ref = ctk.CTkImage(pil, size=(pil.width, pil.height))
        self.cam_preview_label.configure(image=self._snap_ref, text="")
        self.cam_preview_label.pack(pady=4)
        self.cam_status_label.configure(text="✓ Photo captured — analysing…", text_color=C_CYAN)

    def _stop_camera(self):
        self.camera_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.last_frame_bgr = None
        self.start_cam_btn.configure(state="normal")
        self.snap_btn.configure(state="disabled")
        self.stop_cam_btn.configure(state="disabled")
        self.cam_feed_label.configure(image=None, text="Camera stopped")
        self.cam_status_label.configure(text="Camera inactive", text_color=C_DIMTEXT)

    # ══════════════════════════════════════════════════════════════════════
    #  CORE ANALYSIS
    # ══════════════════════════════════════════════════════════════════════

    def _ingest_image(self, img_bgr: np.ndarray):
        """Store image, compute PSD, trigger analysis."""
        self.current_image_bgr = img_bgr
        self.current_psd       = compute_psd(img_bgr)
        self._update_freq_metrics(self.current_psd)
        self._analyse_and_plot()

    def _analyse_and_plot(self):
        if self.current_psd is None:
            return

        denom      = self.denom_var.get()
        golden_psd, golden_thresh = db_load_profile(denom)
        threshold  = self.thresh_var.get() / 100.0

        # ── Plot ──────────────────────────────────────────────────────
        self.ax.clear()
        self._style_ax(self.ax)
        self.ax.plot(self.current_psd, color=C_RED, linewidth=1.3,
                     label="Input Note", alpha=0.9)

        if golden_psd is not None:
            n = min(len(self.current_psd), len(golden_psd))
            self.ax.plot(golden_psd[:n], color=C_CYAN, linewidth=1.3,
                         linestyle="--", label="Golden Profile", alpha=0.9)

        self.ax.legend(facecolor=C_BG3, edgecolor=C_BORDER,
                       labelcolor=C_TEXT, fontsize=9)
        self.fig.tight_layout(pad=1.2)
        self.chart_canvas.draw()

        # ── Verdict ───────────────────────────────────────────────────
        if golden_psd is None:
            self._set_verdict("no_profile", 0.0, denom)
        else:
            sim = pearson_correlation(self.current_psd, golden_psd)
            state = "genuine" if sim >= threshold else "fake"
            self._set_verdict(state, sim, denom)

    def _set_verdict(self, state: str, sim: float, denom: str):
        denom_fmt = f"UGX {denom}"

        if state == "no_profile":
            self.verdict_frame.configure(border_color=C_AMBER)
            self.verdict_icon.configure(text="?", text_color=C_AMBER)
            self.verdict_title.configure(text="No Golden Profile Found", text_color=C_AMBER)
            self.verdict_sub.configure(
                text=f"Save a genuine {denom_fmt} note first, then compare",
                text_color=C_DIMTEXT
            )
            self.verdict_score.configure(text="—", text_color=C_AMBER)
            return

        pct = f"{sim * 100:.1f}%"
        if state == "genuine":
            col = C_GREEN
            self.verdict_frame.configure(border_color="#004d30")
            self.verdict_icon.configure(text="✓", text_color=col)
            self.verdict_title.configure(text="✦  NOTE IS GENUINE", text_color=col)
            self.verdict_sub.configure(
                text=(f"Similarity {pct} — Fourier peaks match the {denom_fmt} "
                      f"golden profile above the {self.thresh_var.get():.0f}% threshold"),
                text_color=C_DIMTEXT
            )
        else:
            col = C_RED
            self.verdict_frame.configure(border_color="#4d001a")
            self.verdict_icon.configure(text="✗", text_color=col)
            self.verdict_title.configure(text="⚠  SUSPECTED COUNTERFEIT", text_color=col)
            self.verdict_sub.configure(
                text=(f"Similarity only {pct} — Fourier peaks deviate significantly "
                      f"from the {denom_fmt} golden profile"),
                text_color=C_DIMTEXT
            )

        self.verdict_score.configure(text=pct, text_color=col)

    def _update_freq_metrics(self, psd: np.ndarray):
        def band_avg(lo, hi):
            arr = psd[lo:hi + 1]
            return f"{arr.mean():.2f}" if len(arr) else "—"

        self._freq_labels["f_low"].configure(text=band_avg(0, 16))
        self._freq_labels["f_mid"].configure(text=band_avg(17, 32))
        self._freq_labels["f_hi"].configure(text=band_avg(33, 50))

    # ══════════════════════════════════════════════════════════════════════
    #  PROFILE MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════

    def save_profile(self):
        if self.current_psd is None:
            messagebox.showerror("Error", "Upload or capture an image first.")
            return
        denom     = self.denom_var.get()
        threshold = self.thresh_var.get() / 100.0
        db_save_profile(denom, self.current_psd, threshold)
        self._refresh_profiles()
        # Re-analyse so the chart updates with the new golden profile
        self._analyse_and_plot()
        # Flash button
        self.save_btn.configure(text="✓  Profile Saved!", text_color=C_GREEN,
                                fg_color="#002a18")
        self.after(2000, lambda: self.save_btn.configure(
            text="★  Save as Golden Profile", text_color=C_GREEN, fg_color=C_BG3))

    def _refresh_profiles(self):
        for widget in self.profiles_frame.winfo_children():
            widget.destroy()

        denoms = db_all_denominations()
        if not denoms:
            ctk.CTkLabel(self.profiles_frame, text="None saved",
                         font=ctk.CTkFont("Courier", 10), text_color=C_DIMTEXT
                         ).pack(anchor="w", padx=8)
            return

        for d in denoms:
            row = ctk.CTkFrame(self.profiles_frame, fg_color="transparent")
            row.pack(fill="x", pady=2)

            ctk.CTkLabel(row, text=f"UGX {d}",
                         font=ctk.CTkFont("Courier", 11), text_color=C_CYAN
                         ).pack(side="left", padx=4)

            ctk.CTkButton(
                row, text="✕", width=24, height=22,
                fg_color="transparent", hover_color="#2a0010",
                text_color=C_RED, border_width=0,
                font=ctk.CTkFont(size=11),
                command=lambda d=d: self._delete_profile(d)
            ).pack(side="right")

    def _delete_profile(self, denom: str):
        if messagebox.askyesno("Delete Profile",
                               f"Delete golden profile for UGX {denom}?"):
            db_delete_profile(denom)
            self._refresh_profiles()
            if self.current_psd is not None:
                self._analyse_and_plot()

    def _clear_all_profiles(self):
        if messagebox.askyesno("Clear All",
                               "Delete ALL stored golden profiles? This cannot be undone."):
            for d in db_all_denominations():
                db_delete_profile(d)
            self._refresh_profiles()
            self._set_verdict("no_profile", 0.0, self.denom_var.get())

    # ══════════════════════════════════════════════════════════════════════
    #  EVENT CALLBACKS
    # ══════════════════════════════════════════════════════════════════════

    def _on_denom_change(self):
        if self.current_psd is not None:
            self._analyse_and_plot()

    def _on_threshold_change(self, value):
        self.thresh_label.configure(text=f"{int(value)}%")
        if self.current_psd is not None:
            self._analyse_and_plot()

    def _on_close(self):
        self._stop_camera()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = SpectralUGX()
    app.mainloop()

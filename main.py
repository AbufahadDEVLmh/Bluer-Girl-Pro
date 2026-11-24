import ctypes
import json
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import customtkinter as ctk

from detector import SmartBodyBlurDetector, apply_blur_to_patch, get_base_dir
try:
    from chrome_ai_server import ChromeAIBridgeServer
except ImportError:
    ChromeAIBridgeServer = None
from video_processor import process_video_file
from overlay import (
    ClickThroughBlurOverlay,
    FastScreenGrabber,
    WDA_EXCLUDEFROMCAPTURE,
    GA_ROOT,
    enable_dpi_awareness,
    get_monitors_list,
    user32,
)

# Enable 1:1 physical pixel DPI awareness before creating windows
enable_dpi_awareness()

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

SETTINGS_FILE = os.path.join(get_base_dir(), "settings.json")


class BluerGirlApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Bluer Girl Pro v6.0 — فزعة الستريمر والشاشة وقراند والأنمي")
        self.geometry("575x840")
        self.minsize(535, 780)
        self.configure(fg_color="#0B0B11")

        # State
        self.is_running = False
        self.panic_full_blur = False
        self.is_window_hidden = False
        self.detector = SmartBodyBlurDetector()
        self.monitors = get_monitors_list()
        self.selected_monitor = self.monitors[0]

        self.blur_mode = "gaussian"
        self.blur_intensity = 88

        self.overlay = ClickThroughBlurOverlay(self, self.selected_monitor)
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._latest_patches_lock = threading.Lock()
        self._latest_patches = None
        self._has_new_frame = False

        # Stats
        self.current_fps = 0.0
        self.current_ms = 0.0
        self.current_boxes_count = 0

        self._build_ui()
        self._load_settings()
        self._setup_control_window_affinity()
        if ChromeAIBridgeServer is not None:
            self.chrome_bridge = ChromeAIBridgeServer(self.detector)
            self.chrome_bridge.start_in_background()
        else:
            self.chrome_bridge = None
        self._start_global_hotkey_poller()
        self._schedule_ui_render_loop()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _setup_control_window_affinity(self):
        """Excludes the control panel itself from screen capture so the AI never scans the app UI."""
        self.update_idletasks()
        try:
            hwnd = user32.GetAncestor(ctypes.wintypes.HWND(self.winfo_id()), GA_ROOT)
            if hwnd:
                user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        except Exception:
            pass

    def _build_ui(self):
        # Header Banner (Najdi Dialect & Premium Streamer Card)
        header = ctk.CTkFrame(
            self,
            fg_color="#151525",
            corner_radius=16,
            border_width=1,
            border_color="#2A3A5C",
        )
        header.pack(fill="x", padx=16, pady=(12, 6))

        title_lbl = ctk.CTkLabel(
            header,
            text="🛡️ Bluer Girl Pro — فزعة الستريمر والشاشة",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#60A5FA",
        )
        title_lbl.pack(pady=(10, 2), padx=14)

        sub_lbl = ctk.CTkLabel(
            header,
            text="يا هلا! أزهله.. أي بنت أو لقطة ما تصلح تطلع بالبث أو بوجهك نغطيها ببلور فوري،\nونفرّق بين الرجال والبنت في قراند والأنمي وتيك توك وكل شي يفتح بالشاشة.",
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#D1D5DB",
            justify="center",
        )
        sub_lbl.pack(pady=(0, 6), padx=14)

        hw_color = "#34D399" if self.detector.is_gpu else "#FBBF24"
        hw_lbl = ctk.CTkLabel(
            header,
            text=f"⚡ شغّال على: {self.detector.hardware_label} (6 نماذج ذكاء اصطناعي)  |  F8: شغّل/وقّف  |  F9: فزعة طوارئ  |  F10: دسّ النافذة",
            font=ctk.CTkFont(family="Segoe UI", size=11, weight="bold"),
            text_color=hw_color,
        )
        hw_lbl.pack(pady=(0, 10), padx=14)

        # Start / Stop Main Action Card
        action_card = ctk.CTkFrame(
            self,
            fg_color="#141420",
            corner_radius=15,
            border_width=1,
            border_color="#222236",
        )
        action_card.pack(fill="x", padx=16, pady=5)

        btn_row = ctk.CTkFrame(action_card, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=(12, 8))

        self.toggle_btn = ctk.CTkButton(
            btn_row,
            text="▶  شغّل البلور الحي عالشاشة الحين (F8)",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            height=46,
            corner_radius=11,
            fg_color="#10B981",
            hover_color="#059669",
            command=self.toggle_live_blur,
        )
        self.toggle_btn.pack(side="right", expand=True, fill="x", padx=(6, 0))

        self.hide_btn = ctk.CTkButton(
            btn_row,
            text="👁️ دسّ النافذة بالخلفية (F10)",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            height=46,
            width=165,
            corner_radius=11,
            fg_color="#252538",
            hover_color="#374151",
            command=self.toggle_window_visibility,
        )
        self.hide_btn.pack(side="left", padx=(0, 6))

        # Quick Preset Profiles Row
        presets_row = ctk.CTkFrame(action_card, fg_color="transparent")
        presets_row.pack(fill="x", padx=14, pady=(0, 8))

        ctk.CTkButton(
            presets_row,
            text="🎥 وضع البث (ستريمر OBS)",
            height=32,
            fg_color="#1E1E32",
            hover_color="#2563EB",
            border_width=1,
            border_color="#3B82F6",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=lambda: self.apply_preset("streamer"),
        ).pack(side="right", expand=True, fill="x", padx=(4, 0))

        ctk.CTkButton(
            presets_row,
            text="🎮 وضع قراند والأنمي",
            height=32,
            fg_color="#1E1E32",
            hover_color="#2563EB",
            border_width=1,
            border_color="#3B82F6",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=lambda: self.apply_preset("balanced"),
        ).pack(side="right", expand=True, fill="x", padx=4)

        ctk.CTkButton(
            presets_row,
            text="👤 غطّ البنت كاملة",
            height=32,
            fg_color="#1E1E32",
            hover_color="#2563EB",
            border_width=1,
            border_color="#3B82F6",
            font=ctk.CTkFont(size=12, weight="bold"),
            command=lambda: self.apply_preset("fullgirl"),
        ).pack(side="left", expand=True, fill="x", padx=(0, 4))

        # Status & Live Telemetry Row
        stats_frame = ctk.CTkFrame(action_card, fg_color="#1B1B2B", corner_radius=9)
        stats_frame.pack(fill="x", padx=14, pady=(0, 10))

        self.status_lbl = ctk.CTkLabel(
            stats_frame,
            text="الحالة: واقف مؤقت  |  اضغط F8 عشان تشغّله أو توقفه بأي وقت وأنت تلعب",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#FBBF24",
        )
        self.status_lbl.pack(pady=6, padx=10)

        # Monitor & Style Card
        mon_card = ctk.CTkFrame(
            self,
            fg_color="#141420",
            corner_radius=15,
            border_width=1,
            border_color="#222236",
        )
        mon_card.pack(fill="x", padx=16, pady=5)

        mon_row = ctk.CTkFrame(mon_card, fg_color="transparent")
        mon_row.pack(fill="x", padx=14, pady=(10, 5))

        ctk.CTkLabel(
            mon_row,
            text="أي شاشة تبي نراقبها؟",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        ).pack(side="right", padx=(6, 0))

        mon_labels = [m["label"] for m in self.monitors]
        self.mon_menu = ctk.CTkOptionMenu(
            mon_row,
            values=mon_labels,
            fg_color="#252538",
            button_color="#3B82F6",
            command=self._on_monitor_change,
            width=265,
        )
        self.mon_menu.set(self.selected_monitor["label"])
        self.mon_menu.pack(side="left")

        style_row = ctk.CTkFrame(mon_card, fg_color="transparent")
        style_row.pack(fill="x", padx=14, pady=5)

        ctk.CTkLabel(
            style_row,
            text="شكل التغبيش (البلور):",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        ).pack(side="right", padx=(6, 0))

        self.style_menu = ctk.CTkOptionMenu(
            style_row,
            values=[
                "تغبيش ضبابي ناعم (Gaussian Blur)",
                "مربعات بكسلة (Mosaic Pixelate)",
                "تغطية سوداء كاتمة (Solid Shield)",
            ],
            fg_color="#252538",
            button_color="#3B82F6",
            command=self._on_style_change,
            width=265,
        )
        self.style_menu.set("تغبيش ضبابي ناعم (Gaussian Blur)")
        self.style_menu.pack(side="left")

        perf_row = ctk.CTkFrame(mon_card, fg_color="transparent")
        perf_row.pack(fill="x", padx=14, pady=(5, 10))

        ctk.CTkLabel(
            perf_row,
            text="قوة اللقط وسرعة الأداء:",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
        ).pack(side="right", padx=(6, 0))

        self.perf_menu = ctk.CTkOptionMenu(
            perf_row,
            values=[
                "قوي وسريع لقراند والأنمي — موصى به (640p)",
                "خفيف مرّة للأجهزة الضعيفة (480p)",
                "دقّة خارقة للبعيد والتفاصيل الصغيرة (800p)",
            ],
            fg_color="#252538",
            button_color="#3B82F6",
            command=self._on_perf_change,
            width=265,
        )
        self.perf_menu.set("قوي وسريع لقراند والأنمي — موصى به (640p)")
        self.perf_menu.pack(side="left")

        # Targeted Body Parts Card
        parts_card = ctk.CTkFrame(
            self,
            fg_color="#141420",
            corner_radius=15,
            border_width=1,
            border_color="#222236",
        )
        parts_card.pack(fill="x", padx=16, pady=5)

        ctk.CTkLabel(
            parts_card,
            text="🎯 وش تبي نغطي بالضبط؟ (الرجال والعيال ما عليهم بلور أبد)",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="bold"),
            text_color="#60A5FA",
        ).pack(anchor="e", padx=14, pady=(8, 4))

        grid_frame = ctk.CTkFrame(parts_card, fg_color="transparent")
        grid_frame.pack(fill="x", padx=14, pady=(2, 8))

        self.var_face = tk.BooleanVar(value=True)
        self.var_hair = tk.BooleanVar(value=True)
        self.var_chest = tk.BooleanVar(value=True)
        self.var_limbs = tk.BooleanVar(value=True)
        self.var_fullbody = tk.BooleanVar(value=False)
        self.var_anime_games = tk.BooleanVar(value=True)
        self.var_obs_mode = tk.BooleanVar(value=False)

        cb1 = ctk.CTkCheckBox(
            grid_frame,
            text="وجه البنت بس (العيال يتخطاهم)",
            variable=self.var_face,
            command=self._sync_detector_flags,
            font=ctk.CTkFont(family="Segoe UI", size=13),
        )
        cb1.grid(row=0, column=1, sticky="e", padx=8, pady=4)

        cb2 = ctk.CTkCheckBox(
            grid_frame,
            text="الشعر والراس",
            variable=self.var_hair,
            command=self._sync_detector_flags,
            font=ctk.CTkFont(family="Segoe UI", size=13),
        )
        cb2.grid(row=0, column=0, sticky="e", padx=8, pady=4)

        cb3 = ctk.CTkCheckBox(
            grid_frame,
            text="الصدر والنحر والبطن والأماكن الفاتنة",
            variable=self.var_chest,
            command=self._sync_detector_flags,
            font=ctk.CTkFont(family="Segoe UI", size=13),
        )
        cb3.grid(row=1, column=1, sticky="e", padx=8, pady=4)

        cb4 = ctk.CTkCheckBox(
            grid_frame,
            text="الذرعان والسيقان والأكتاف المكشوفة",
            variable=self.var_limbs,
            command=self._sync_detector_flags,
            font=ctk.CTkFont(family="Segoe UI", size=13),
        )
        cb4.grid(row=1, column=0, sticky="e", padx=8, pady=4)

        cb_anime = ctk.CTkCheckBox(
            grid_frame,
            text="🎮 ضبط قراند والألعاب والأنمي (يموّه البنات بس ويخلّي العيال)",
            variable=self.var_anime_games,
            command=self._sync_detector_flags,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#34D399",
        )
        cb_anime.grid(row=2, column=0, columnspan=2, sticky="e", padx=8, pady=(5, 2))

        cb_obs = ctk.CTkCheckBox(
            grid_frame,
            text="🎥 وضع الستريمر (OBS): خلّ البلور يطلع للمتابعين بالبث وتصوير الشاشة",
            variable=self.var_obs_mode,
            command=self._sync_detector_flags,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            text_color="#FBBF24",
        )
        cb_obs.grid(row=3, column=0, columnspan=2, sticky="e", padx=8, pady=(3, 2))

        cb5 = ctk.CTkCheckBox(
            grid_frame,
            text="غطّ البنت كاملة من راسها لرجليها لا طلعت بالشاشة",
            variable=self.var_fullbody,
            command=self._sync_detector_flags,
            font=ctk.CTkFont(family="Segoe UI", size=12),
            text_color="#D1D5DB",
        )
        cb5.grid(row=4, column=0, columnspan=2, sticky="e", padx=8, pady=(3, 2))
        grid_frame.columnconfigure(0, weight=1)
        grid_frame.columnconfigure(1, weight=1)

        # Sliders Card
        sliders_card = ctk.CTkFrame(
            self,
            fg_color="#141420",
            corner_radius=15,
            border_width=1,
            border_color="#222236",
        )
        sliders_card.pack(fill="x", padx=16, pady=5)

        # 1. Blur Strength
        s1_top = ctk.CTkFrame(sliders_card, fg_color="transparent")
        s1_top.pack(fill="x", padx=14, pady=(8, 0))
        ctk.CTkLabel(
            s1_top,
            text="قوة التغبيش (ثقل البلور):",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="right")
        self.lbl_intensity = ctk.CTkLabel(
            s1_top,
            text="88%",
            text_color="#34D399",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.lbl_intensity.pack(side="left")

        self.slider_intensity = ctk.CTkSlider(
            sliders_card,
            from_=25,
            to=100,
            number_of_steps=75,
            command=self._on_intensity_slide,
        )
        self.slider_intensity.set(88)
        self.slider_intensity.pack(fill="x", padx=14, pady=(2, 6))

        # 2. Padding / Expansion
        s2_top = ctk.CTkFrame(sliders_card, fg_color="transparent")
        s2_top.pack(fill="x", padx=14, pady=(2, 0))
        ctk.CTkLabel(
            s2_top,
            text="وسّع التغطية شوي حول المكان المكشوف:",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="right")
        self.lbl_padding = ctk.CTkLabel(
            s2_top,
            text="10%",
            text_color="#34D399",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.lbl_padding.pack(side="left")

        self.slider_padding = ctk.CTkSlider(
            sliders_card,
            from_=0,
            to=35,
            number_of_steps=35,
            command=self._on_padding_slide,
        )
        self.slider_padding.set(10)
        self.slider_padding.pack(fill="x", padx=14, pady=(2, 6))

        # 3. Detection Sensitivity
        s3_top = ctk.CTkFrame(sliders_card, fg_color="transparent")
        s3_top.pack(fill="x", padx=14, pady=(2, 0))
        ctk.CTkLabel(
            s3_top,
            text="دقّة اللقط وحساسية الكشف:",
            font=ctk.CTkFont(size=12, weight="bold"),
        ).pack(side="right")
        self.lbl_sens = ctk.CTkLabel(
            s3_top,
            text="مضبوطة (75%)",
            text_color="#34D399",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.lbl_sens.pack(side="left")

        self.slider_sens = ctk.CTkSlider(
            sliders_card,
            from_=50,
            to=90,
            number_of_steps=40,
            command=self._on_sens_slide,
        )
        self.slider_sens.set(75)
        self.slider_sens.pack(fill="x", padx=14, pady=(2, 10))

        # Bottom Utility Buttons
        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.pack(fill="x", padx=16, pady=(4, 10))

        test_btn = ctk.CTkButton(
            bottom_frame,
            text="🖼️ جرّب على صورة",
            fg_color="#252538",
            hover_color="#374151",
            height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.test_on_image_file,
        )
        test_btn.pack(side="left", expand=True, fill="x", padx=(0, 4))

        vid_btn = ctk.CTkButton(
            bottom_frame,
            text="🎬 غبّش ملف فيديو (.MP4)",
            fg_color="#1E3A8A",
            hover_color="#2563EB",
            height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.process_video_dialog,
        )
        vid_btn.pack(side="left", expand=True, fill="x", padx=4)

        panic_btn = ctk.CTkButton(
            bottom_frame,
            text="🔒 فزعة طوارئ (F9)",
            fg_color="#7F1D1D",
            hover_color="#991B1B",
            height=36,
            font=ctk.CTkFont(size=12, weight="bold"),
            command=self.toggle_panic_blur,
        )
        panic_btn.pack(side="right", expand=True, fill="x", padx=(4, 0))

    def apply_preset(self, mode: str):
        if mode == "streamer":
            self.var_face.set(True)
            self.var_hair.set(True)
            self.var_chest.set(True)
            self.var_limbs.set(True)
            self.var_anime_games.set(True)
            self.var_obs_mode.set(True)
            self.var_fullbody.set(False)
            self.slider_intensity.set(92)
            self._on_intensity_slide(92)
            self.slider_padding.set(14)
            self._on_padding_slide(14)
            self.slider_sens.set(80)
            self._on_sens_slide(80)
        elif mode == "fullgirl":
            self.var_face.set(True)
            self.var_hair.set(True)
            self.var_chest.set(True)
            self.var_limbs.set(True)
            self.var_anime_games.set(True)
            self.var_fullbody.set(True)
            self.slider_intensity.set(94)
            self._on_intensity_slide(94)
            self.slider_padding.set(15)
            self._on_padding_slide(15)
            self.slider_sens.set(78)
            self._on_sens_slide(78)
        else:
            self.var_face.set(True)
            self.var_hair.set(True)
            self.var_chest.set(True)
            self.var_limbs.set(True)
            self.var_anime_games.set(True)
            self.var_fullbody.set(False)
            self.slider_intensity.set(88)
            self._on_intensity_slide(88)
            self.slider_padding.set(10)
            self._on_padding_slide(10)
            self.slider_sens.set(75)
            self._on_sens_slide(75)
        self._sync_detector_flags(save=True)

    def _save_settings(self):
        data = {
            "blur_mode": self.blur_mode,
            "blur_intensity": self.blur_intensity,
            "padding_pct": int(self.slider_padding.get()),
            "sens_pct": int(self.slider_sens.get()),
            "performance_mode": self.detector.performance_mode,
            "blur_face": bool(self.var_face.get()),
            "blur_hair": bool(self.var_hair.get()),
            "blur_chest": bool(self.var_chest.get()),
            "blur_limbs": bool(self.var_limbs.get()),
            "blur_fullbody": bool(self.var_fullbody.get()),
            "enable_anime_games": bool(self.var_anime_games.get()),
            "obs_mode": bool(self.var_obs_mode.get()),
        }
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_settings(self):
        if not os.path.exists(SETTINGS_FILE):
            return
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.var_face.set(bool(data.get("blur_face", True)))
            self.var_hair.set(bool(data.get("blur_hair", True)))
            self.var_chest.set(bool(data.get("blur_chest", True)))
            self.var_limbs.set(bool(data.get("blur_limbs", True)))
            self.var_fullbody.set(bool(data.get("blur_fullbody", False)))
            self.var_anime_games.set(bool(data.get("enable_anime_games", True)))
            self.var_obs_mode.set(bool(data.get("obs_mode", False)))

            intensity = int(data.get("blur_intensity", 88))
            self.slider_intensity.set(intensity)
            self._on_intensity_slide(intensity)

            pad_pct = int(data.get("padding_pct", 10))
            self.slider_padding.set(pad_pct)
            self._on_padding_slide(pad_pct)

            sens_pct = int(data.get("sens_pct", 75))
            self.slider_sens.set(sens_pct)
            self._on_sens_slide(sens_pct)

            bmode = data.get("blur_mode", "gaussian")
            self.blur_mode = bmode
            if bmode == "pixelate":
                self.style_menu.set("مربعات بكسلة (Mosaic Pixelate)")
            elif bmode == "solid":
                self.style_menu.set("تغطية سوداء كاتمة (Solid Shield)")
            else:
                self.style_menu.set("تغبيش ضبابي ناعم (Gaussian Blur)")

            pmode = data.get("performance_mode", "balanced")
            self.detector.set_performance_mode(pmode)
            if pmode == "light":
                self.perf_menu.set("خفيف مرّة للأجهزة الضعيفة (480p)")
            elif pmode == "high":
                self.perf_menu.set("دقّة خارقة للبعيد والتفاصيل الصغيرة (800p)")
            else:
                self.perf_menu.set("قوي وسريع لقراند والأنمي — موصى به (640p)")

            self._sync_detector_flags(save=False)
        except Exception:
            pass

    def _sync_detector_flags(self, save: bool = True):
        self.detector.blur_face = bool(self.var_face.get())
        self.detector.blur_hair = bool(self.var_hair.get())
        self.detector.blur_chest_attractive = bool(self.var_chest.get())
        self.detector.blur_arms_legs = bool(self.var_limbs.get())
        self.detector.blur_full_body = bool(self.var_fullbody.get())
        self.detector.enable_anime_games = bool(self.var_anime_games.get())
        self.overlay.set_obs_capture_visible(bool(self.var_obs_mode.get()))
        self.detector.tracker.clear()
        if save:
            self._save_settings()

    def _on_monitor_change(self, selected_label: str):
        for m in self.monitors:
            if m["label"] == selected_label:
                self.selected_monitor = m
                break
        self.overlay.set_monitor(self.selected_monitor)
        self.detector.tracker.clear()

    def _on_style_change(self, val: str):
        if "بكسلة" in val:
            self.blur_mode = "pixelate"
        elif "كاتمة" in val or "معتمة" in val:
            self.blur_mode = "solid"
        else:
            self.blur_mode = "gaussian"
        self._save_settings()

    def _on_perf_change(self, val: str):
        if "480p" in val:
            self.detector.set_performance_mode("light")
        elif "800p" in val:
            self.detector.set_performance_mode("high")
        else:
            self.detector.set_performance_mode("balanced")
        self._save_settings()

    def _on_intensity_slide(self, val):
        self.blur_intensity = int(val)
        self.lbl_intensity.configure(text=f"{self.blur_intensity}%")
        self._save_settings()

    def _on_padding_slide(self, val):
        pct = int(val)
        self.detector.box_padding_ratio = pct / 100.0
        self.lbl_padding.configure(text=f"{pct}%")
        self._save_settings()

    def _on_sens_slide(self, val):
        sens = int(val)
        thresh = 0.44 - ((sens - 50) / 40.0) * 0.20
        self.detector.confidence_threshold = round(max(0.24, min(0.45, thresh)), 2)
        level = "خارقة" if sens >= 82 else ("مضبوطة" if sens >= 70 else "هادية")
        self.lbl_sens.configure(text=f"{level} ({sens}%)")
        self._save_settings()

    def toggle_window_visibility(self):
        if self.is_window_hidden:
            self.deiconify()
            self.lift()
            self.is_window_hidden = False
        else:
            if not self.is_running:
                self.start_live_blur()
            self.withdraw()
            self.is_window_hidden = True

    def toggle_live_blur(self):
        if self.is_running:
            self.stop_live_blur()
        else:
            self.start_live_blur()

    def start_live_blur(self):
        if self.is_running:
            return
        self.is_running = True
        self._stop_event.clear()
        self.detector.tracker.clear()
        self.detector.gender_memory.clear()
        self.toggle_btn.configure(
            text="⏹  وقّف البلور الحي مؤقت (F8)",
            fg_color="#EF4444",
            hover_color="#DC2626",
        )
        self.status_lbl.configure(
            text="الحالة: شغّال الحين ويراقب الشاشة...",
            text_color="#34D399",
        )
        self._worker_thread = threading.Thread(
            target=self._capture_and_detect_loop, daemon=True
        )
        self._worker_thread.start()

    def stop_live_blur(self):
        self.is_running = False
        self.panic_full_blur = False
        self._stop_event.set()
        with self._latest_patches_lock:
            self._latest_patches = []
            self._has_new_frame = True
        self.overlay.clear_overlay()
        self.toggle_btn.configure(
            text="▶  شغّل البلور الحي عالشاشة الحين (F8)",
            fg_color="#10B981",
            hover_color="#059669",
        )
        self.status_lbl.configure(
            text="الحالة: واقف مؤقت  |  اضغط F8 عشان تشغّله أو توقفه بأي وقت",
            text_color="#FBBF24",
        )

    def toggle_panic_blur(self):
        self.panic_full_blur = not self.panic_full_blur
        if self.panic_full_blur and not self.is_running:
            self.start_live_blur()

    def _capture_and_detect_loop(self):
        mon = dict(self.selected_monitor)
        grabber = FastScreenGrabber(mon["left"], mon["top"], mon["width"], mon["height"])

        try:
            while not self._stop_event.is_set():
                t_start = time.perf_counter()

                if (
                    mon["left"] != self.selected_monitor["left"]
                    or mon["top"] != self.selected_monitor["top"]
                    or mon["width"] != self.selected_monitor["width"]
                    or mon["height"] != self.selected_monitor["height"]
                ):
                    grabber.close()
                    mon = dict(self.selected_monitor)
                    grabber = FastScreenGrabber(
                        mon["left"], mon["top"], mon["width"], mon["height"]
                    )

                bgr_frame = grabber.grab_bgr()
                if bgr_frame is None:
                    time.sleep(0.03)
                    continue

                fh, fw = bgr_frame.shape[:2]

                if self.panic_full_blur:
                    boxes = [[0, 0, fw, fh]]
                else:
                    boxes = self.detector.detect_frame(bgr_frame, use_tracker=True)

                patches = []
                mode = self.blur_mode
                intensity = self.blur_intensity

                for x1, y1, x2, y2 in boxes:
                    cx1 = max(0, min(fw - 1, int(x1)))
                    cy1 = max(0, min(fh - 1, int(y1)))
                    cx2 = max(cx1 + 2, min(fw, int(x2)))
                    cy2 = max(cy1 + 2, min(fh, int(y2)))

                    crop = bgr_frame[cy1:cy2, cx1:cx2]
                    if crop.size == 0:
                        continue
                    blurred_bgr = apply_blur_to_patch(
                        crop, mode=mode, intensity=intensity
                    )
                    blurred_rgb = cv2.cvtColor(blurred_bgr, cv2.COLOR_BGR2RGB)
                    patches.append((cx1, cy1, cx2, cy2, blurred_rgb))

                elapsed = max(0.001, time.perf_counter() - t_start)
                self.current_fps = 1.0 / elapsed
                self.current_ms = self.detector.last_inference_ms
                self.current_boxes_count = len(patches)

                with self._latest_patches_lock:
                    self._latest_patches = patches
                    self._has_new_frame = True

                sleep_time = max(0.004, 0.025 - elapsed)
                time.sleep(sleep_time)
        finally:
            grabber.close()

    def _schedule_ui_render_loop(self):
        if self.is_running:
            patches_to_draw = None
            with self._latest_patches_lock:
                if self._has_new_frame:
                    patches_to_draw = self._latest_patches
                    self._has_new_frame = False

            if patches_to_draw is not None:
                self.overlay.render_patches(patches_to_draw)

            if not self.is_window_hidden:
                state_txt = (
                    f"شغّال الحين 🟢  |  الأماكن المغطاة: {self.current_boxes_count}  |  "
                    f"سرعة اللقط: {self.current_ms:.1f}ms  |  الفريمات: {self.current_fps:.0f} FPS"
                )
                self.status_lbl.configure(text=state_txt, text_color="#34D399")
        elif not self.is_window_hidden and hasattr(self, "chrome_bridge"):
            reqs = self.chrome_bridge.total_requests
            if reqs > 0:
                hits = self.chrome_bridge.cache_hits
                b_cnt = self.chrome_bridge.total_blurred_detections
                avg_m = self.chrome_bridge.avg_ms
                self.status_lbl.configure(
                    text=f"🌐 إضافة كروم شغّالة  |  فحصنا: {reqs} (كاش: {hits})  |  غطّينا: {b_cnt}  |  {avg_m:.1f}ms",
                    text_color="#60A5FA",
                )

        self.after(16, self._schedule_ui_render_loop)

    def _start_global_hotkey_poller(self):
        self._f8_down = False
        self._f9_down = False
        self._f10_down = False

        def poll_keys():
            try:
                f8_state = user32.GetAsyncKeyState(0x77) & 0x8000
                if f8_state and not self._f8_down:
                    self._f8_down = True
                    self.toggle_live_blur()
                elif not f8_state:
                    self._f8_down = False

                f9_state = user32.GetAsyncKeyState(0x78) & 0x8000
                if f9_state and not self._f9_down:
                    self._f9_down = True
                    self.toggle_panic_blur()
                elif not f9_state:
                    self._f9_down = False

                f10_state = user32.GetAsyncKeyState(0x79) & 0x8000
                if f10_state and not self._f10_down:
                    self._f10_down = True
                    self.toggle_window_visibility()
                elif not f10_state:
                    self._f10_down = False
            except Exception:
                pass
            self.after(70, poll_keys)

        self.after(100, poll_keys)

    def test_on_image_file(self):
        file_path = filedialog.askopenfilename(
            title="اختر صورة تبي تجرّب التغبيش الذكي عليها",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")],
        )
        if not file_path or not os.path.exists(file_path):
            return

        raw_bytes = np.fromfile(file_path, dtype=np.uint8)
        bgr = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            return

        self.detector.gender_memory.clear()
        boxes = self.detector.detect_frame(bgr, use_tracker=False)
        result_bgr = bgr.copy()
        fh, fw = result_bgr.shape[:2]

        for x1, y1, x2, y2 in boxes:
            cx1, cy1 = max(0, x1), max(0, y1)
            cx2, cy2 = min(fw, x2), min(fh, y2)
            if cx2 > cx1 + 2 and cy2 > cy1 + 2:
                result_bgr[cy1:cy2, cx1:cx2] = apply_blur_to_patch(
                    result_bgr[cy1:cy2, cx1:cx2],
                    mode=self.blur_mode,
                    intensity=self.blur_intensity,
                )

        preview_win = ctk.CTkToplevel(self)
        preview_win.title(f"معاينة البلور — غطّينا لك {len(boxes)} مكان مكشوف")
        preview_win.geometry("820x620")
        preview_win.attributes("-topmost", True)

        rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((780, 540), Image.Resampling.LANCZOS)
        ctk_img = ctk.CTkImage(
            light_image=pil_img, dark_image=pil_img, size=pil_img.size
        )

        lbl = ctk.CTkLabel(preview_win, text="", image=ctk_img)
        lbl.pack(expand=True, padx=16, pady=12)

        info = ctk.CTkLabel(
            preview_win,
            text=f"كفو! غطّينا {len(boxes)} مكان مكشوف في {self.detector.last_inference_ms:.1f} ملي ثانية بس",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#34D399",
        )
        info.pack(pady=(0, 12))

    def process_video_dialog(self):
        in_path = filedialog.askopenfilename(
            title="اختر مقطع فيديو تبي نغبّش البنات اللي فيه ونحفظه لك",
            filetypes=[("Video Files", "*.mp4 *.mkv *.mov *.avi *.webm")],
        )
        if not in_path or not os.path.exists(in_path):
            return

        base, _ = os.path.splitext(in_path)
        default_out = f"{base}_blurred.mp4"
        out_path = filedialog.asksaveasfilename(
            title="وين تبي نحفظ المقطع بعد التغبيش؟",
            initialfile=os.path.basename(default_out),
            defaultextension=".mp4",
            filetypes=[("MP4 Video", "*.mp4")],
        )
        if not out_path:
            return

        prog_win = ctk.CTkToplevel(self)
        prog_win.title("🎬 جاري تغبيش المقطع وتجهيزه...")
        prog_win.geometry("460x210")
        prog_win.attributes("-topmost", True)

        lbl_status = ctk.CTkLabel(
            prog_win,
            text="أزهله.. قاعدين نفحص ونغبّش فريمات المقطع بكرت الشاشة...",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        lbl_status.pack(pady=(18, 8), padx=16)

        pbar = ctk.CTkProgressBar(prog_win, width=400)
        pbar.set(0.0)
        pbar.pack(pady=8, padx=16)

        lbl_detail = ctk.CTkLabel(
            prog_win,
            text="0%  |  الفريم: 0 / 0",
            font=ctk.CTkFont(size=12),
            text_color="#9CA3AF",
        )
        lbl_detail.pack(pady=4)

        cancel_ev = threading.Event()
        btn_cancel = ctk.CTkButton(
            prog_win,
            text="كنسل المعالجة",
            fg_color="#EF4444",
            hover_color="#DC2626",
            height=32,
            command=cancel_ev.set,
        )
        btn_cancel.pack(pady=(8, 12))

        def on_prog(f_idx, total_f, pct, fps_speed):
            def ui_up():
                if not prog_win.winfo_exists():
                    return
                pbar.set(pct / 100.0)
                lbl_detail.configure(
                    text=f"{pct:.1f}%  |  الفريم: {f_idx} / {total_f}  |  السرعة: {fps_speed:.1f} FPS"
                )

            self.after(0, ui_up)

        def worker():
            try:
                res = process_video_file(
                    in_path,
                    out_path,
                    self.detector,
                    blur_mode=self.blur_mode,
                    blur_intensity=self.blur_intensity,
                    progress_callback=on_prog,
                    cancel_event=cancel_ev,
                )

                def done_ui():
                    if not prog_win.winfo_exists():
                        return
                    if res.get("cancelled"):
                        lbl_status.configure(
                            text="خلاص كنسلنا معالجة المقطع.", text_color="#FBBF24"
                        )
                    else:
                        pbar.set(1.0)
                        lbl_status.configure(
                            text=f"✅ خلصنا المقطع وحفظناه لك في {res.get('seconds')} ثانية!",
                            text_color="#34D399",
                        )
                        lbl_detail.configure(
                            text=f"غبّشنا لك {res.get('blurred_frames')} فريم من أصل {res.get('frames_processed')}"
                        )
                        btn_cancel.configure(
                            text="تم، قفّل النافذة",
                            fg_color="#10B981",
                            command=prog_win.destroy,
                        )

                self.after(0, done_ui)
            except Exception as e:
                self.after(
                    0,
                    lambda: lbl_status.configure(
                        text=f"صار خطأ: {e}", text_color="#EF4444"
                    ),
                )

        threading.Thread(target=worker, daemon=True).start()

    def on_close(self):
        self._save_settings()
        self.stop_live_blur()
        self.overlay.destroy()
        self.destroy()


if __name__ == "__main__":
    app = BluerGirlApp()
    app.mainloop()

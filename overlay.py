import ctypes
from ctypes import wintypes
import tkinter as tk
import cv2
import numpy as np
from PIL import Image, ImageTk
import mss


# Win32 Constants
GWL_EXSTYLE = -20
WS_EX_LAYERED = 0x00080000
WS_EX_TRANSPARENT = 0x00000020
WS_EX_TOPMOST = 0x00000008
WS_EX_TOOLWINDOW = 0x00000080
WS_EX_NOACTIVATE = 0x08000000

LWA_ALPHA = 0x00000002
WDA_EXCLUDEFROMCAPTURE = 0x00000011
GA_ROOT = 2

HWND_TOPMOST = wintypes.HWND(-1)
HWND_TOP = wintypes.HWND(0)
SWP_NOACTIVATE = 0x0010
SWP_SHOWWINDOW = 0x0040
SWP_HIDEWINDOW = 0x0080
SWP_NOOWNERZORDER = 0x0200

SRCCOPY = 0x00CC0020  # Captures screen without layered overlay windows
DIB_RGB_COLORS = 0
BI_RGB = 0

user32 = ctypes.windll.user32
gdi32 = ctypes.windll.gdi32

# Proper 64-bit Win32 Function Signatures (CRITICAL for 64-bit Python on Windows)
user32.GetDesktopWindow.argtypes = []
user32.GetDesktopWindow.restype = wintypes.HWND

user32.GetWindowDC.argtypes = [wintypes.HWND]
user32.GetWindowDC.restype = wintypes.HDC

user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.ReleaseDC.restype = ctypes.c_int

user32.GetAncestor.argtypes = [wintypes.HWND, ctypes.c_uint]
user32.GetAncestor.restype = wintypes.HWND

user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.GetWindowLongW.restype = ctypes.c_long

user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_long]
user32.SetWindowLongW.restype = ctypes.c_long

user32.SetLayeredWindowAttributes.argtypes = [
    wintypes.HWND,
    wintypes.COLORREF,
    wintypes.BYTE,
    wintypes.DWORD,
]
user32.SetLayeredWindowAttributes.restype = wintypes.BOOL

user32.SetWindowDisplayAffinity.argtypes = [wintypes.HWND, wintypes.DWORD]
user32.SetWindowDisplayAffinity.restype = wintypes.BOOL

user32.SetWindowPos.argtypes = [
    wintypes.HWND,
    wintypes.HWND,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_uint,
]
user32.SetWindowPos.restype = wintypes.BOOL

user32.SetWindowRgn.argtypes = [wintypes.HWND, wintypes.HRGN, wintypes.BOOL]
user32.SetWindowRgn.restype = ctypes.c_int

user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.IsWindowVisible.restype = wintypes.BOOL

gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC

gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP

gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ

gdi32.BitBlt.argtypes = [
    wintypes.HDC,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.HDC,
    ctypes.c_int,
    ctypes.c_int,
    wintypes.DWORD,
]
gdi32.BitBlt.restype = wintypes.BOOL

gdi32.GetDIBits.argtypes = [
    wintypes.HDC,
    wintypes.HBITMAP,
    ctypes.c_uint,
    ctypes.c_uint,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint,
]
gdi32.GetDIBits.restype = ctypes.c_int

gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteObject.restype = wintypes.BOOL

gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.DeleteDC.restype = wintypes.BOOL

gdi32.CreateRoundRectRgn.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
]
gdi32.CreateRoundRectRgn.restype = wintypes.HRGN


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]


class BITMAPINFO(ctypes.Structure):
    _fields_ = [
        ("bmiHeader", BITMAPINFOHEADER),
        ("bmiColors", wintypes.DWORD * 3),
    ]


def enable_dpi_awareness():
    """Ensures 1:1 physical pixel coordinates across all monitors on Windows."""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass


def get_monitors_list():
    """
    Returns list of monitor dicts:
    [{'id': i, 'label': str, 'left': int, 'top': int, 'width': int, 'height': int}, ...]
    """
    monitors = []
    try:
        with mss.MSS() as sct:
            raw_mons = sct.monitors
            for idx in range(1, len(raw_mons)):
                m = raw_mons[idx]
                primary_tag = " (الرئيسية)" if m.get("is_primary") or idx == 1 else ""
                monitors.append(
                    {
                        "id": idx,
                        "label": f"الشاشة {idx}{primary_tag} — {m['width']}x{m['height']}",
                        "left": int(m["left"]),
                        "top": int(m["top"]),
                        "width": int(m["width"]),
                        "height": int(m["height"]),
                    }
                )
            if len(raw_mons) > 2:
                m0 = raw_mons[0]
                monitors.append(
                    {
                        "id": 0,
                        "label": f"جميع الشاشات معاً — {m0['width']}x{m0['height']}",
                        "left": int(m0["left"]),
                        "top": int(m0["top"]),
                        "width": int(m0["width"]),
                        "height": int(m0["height"]),
                    }
                )
    except Exception:
        w = user32.GetSystemMetrics(0)
        h = user32.GetSystemMetrics(1)
        monitors.append(
            {
                "id": 1,
                "label": f"الشاشة الرئيسية — {w}x{h}",
                "left": 0,
                "top": 0,
                "width": w,
                "height": h,
            }
        )
    return monitors


class FastScreenGrabber:
    """
    Fast 64-bit Win32 GDI + MSS screen capture.
    """

    def __init__(self, left: int, top: int, width: int, height: int):
        self.left = left
        self.top = top
        self.width = width
        self.height = height

        self.hdesktop = user32.GetDesktopWindow()
        self.srcdc = None
        self.memdc = None
        self.bmp = None
        self.bmi = BITMAPINFO()
        self.bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        self.bmi.bmiHeader.biWidth = width
        self.bmi.bmiHeader.biHeight = -height  # top-down image
        self.bmi.bmiHeader.biPlanes = 1
        self.bmi.bmiHeader.biBitCount = 32
        self.bmi.bmiHeader.biCompression = BI_RGB

        self.buf_len = width * height * 4
        self.buffer = ctypes.create_string_buffer(self.buf_len)
        self._init_gdi()
        self._mss = None

    def _init_gdi(self):
        self.cleanup_gdi()
        self.srcdc = user32.GetWindowDC(self.hdesktop)
        if self.srcdc:
            self.memdc = gdi32.CreateCompatibleDC(self.srcdc)
            self.bmp = gdi32.CreateCompatibleBitmap(self.srcdc, self.width, self.height)
            if self.memdc and self.bmp:
                gdi32.SelectObject(self.memdc, self.bmp)

    def grab_bgr(self) -> np.ndarray:
        """Captures the monitor and returns a contiguous BGR uint8 array (H, W, 3)."""
        if self.srcdc and self.memdc and self.bmp:
            ok = gdi32.BitBlt(
                self.memdc,
                0,
                0,
                self.width,
                self.height,
                self.srcdc,
                self.left,
                self.top,
                SRCCOPY,
            )
            if ok:
                lines = gdi32.GetDIBits(
                    self.memdc,
                    self.bmp,
                    0,
                    self.height,
                    self.buffer,
                    ctypes.byref(self.bmi),
                    DIB_RGB_COLORS,
                )
                if lines == self.height:
                    img_bgra = np.frombuffer(self.buffer, dtype=np.uint8).reshape(
                        (self.height, self.width, 4)
                    )
                    return cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

        try:
            if self._mss is None:
                self._mss = mss.MSS()
            sct_img = self._mss.grab(
                {
                    "left": self.left,
                    "top": self.top,
                    "width": self.width,
                    "height": self.height,
                }
            )
            img_bgra = np.array(sct_img, dtype=np.uint8)
            return cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)
        except Exception:
            self._init_gdi()
            return None

    def cleanup_gdi(self):
        try:
            if self.bmp:
                gdi32.DeleteObject(self.bmp)
                self.bmp = None
            if self.memdc:
                gdi32.DeleteDC(self.memdc)
                self.memdc = None
            if self.srcdc:
                user32.ReleaseDC(self.hdesktop, self.srcdc)
                self.srcdc = None
        except Exception:
            pass

    def close(self):
        self.cleanup_gdi()
        if self._mss is not None:
            try:
                self._mss.close()
            except Exception:
                pass
            self._mss = None


class MiniPatchWindow:
    """
    A single small click-through popup window sized strictly to one detected body part.
    Never covers the whole screen — only exists at the exact [x, y, w, h] of the blurred part.
    """

    def __init__(self, master: tk.Misc):
        self.win = tk.Toplevel(master)
        self.win.overrideredirect(True)
        self.win.geometry("2x2+-10000+-10000")
        self.win.configure(bg="#18161C")
        self.win.wm_attributes("-topmost", True)

        self.canvas = tk.Canvas(
            self.win,
            bg="#18161C",
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.img_item = self.canvas.create_image(0, 0, anchor="nw")
        self.photo_ref = None

        # Must deiconify & update once so Tkinter creates the real Win32 wrapper HWND
        self.win.deiconify()
        self.win.update_idletasks()

        self.hwnd = user32.GetAncestor(wintypes.HWND(self.win.winfo_id()), GA_ROOT)
        if not self.hwnd:
            self.hwnd = wintypes.HWND(
                int(self.win.wm_frame(), 16)
                if self.win.wm_frame() != "0x0"
                else self.win.winfo_id()
            )

        self._apply_win32_styles()
        # Hide immediately after configuring Win32 styles
        user32.SetWindowPos(
            self.hwnd,
            HWND_TOP,
            0,
            0,
            0,
            0,
            SWP_NOACTIVATE | SWP_NOOWNERZORDER | SWP_HIDEWINDOW,
        )
        self.is_visible = False

    def _apply_win32_styles(self):
        ex_style = user32.GetWindowLongW(self.hwnd, GWL_EXSTYLE)
        new_ex = (
            ex_style
            | WS_EX_LAYERED
            | WS_EX_TRANSPARENT
            | WS_EX_TOPMOST
            | WS_EX_TOOLWINDOW
            | WS_EX_NOACTIVATE
        )
        user32.SetWindowLongW(self.hwnd, GWL_EXSTYLE, new_ex)
        user32.SetLayeredWindowAttributes(self.hwnd, 0, 255, LWA_ALPHA)
        user32.SetWindowDisplayAffinity(self.hwnd, WDA_EXCLUDEFROMCAPTURE)

    def set_obs_capture_visible(self, visible: bool):
        """
        If visible=True (Streamer OBS Mode), sets WDA_NONE (0x00) so OBS/Streamlabs/Discord
        captures the blur overlay on stream. If False (Personal Mode), sets WDA_EXCLUDEFROMCAPTURE (0x11).
        """
        affinity = 0x00000000 if visible else WDA_EXCLUDEFROMCAPTURE
        user32.SetWindowDisplayAffinity(self.hwnd, affinity)

    def show_patch(
        self, screen_x: int, screen_y: int, w: int, h: int, rgb_patch: np.ndarray
    ):
        if w <= 2 or h <= 2:
            self.hide()
            return

        pil_img = Image.fromarray(rgb_patch)
        self.photo_ref = ImageTk.PhotoImage(pil_img)
        self.canvas.config(width=w, height=h)
        self.canvas.itemconfigure(self.img_item, image=self.photo_ref)

        flags = SWP_NOACTIVATE | SWP_NOOWNERZORDER | SWP_SHOWWINDOW
        user32.SetWindowPos(
            self.hwnd,
            HWND_TOPMOST,
            int(screen_x),
            int(screen_y),
            int(w),
            int(h),
            flags,
        )
        self.is_visible = True

    def hide(self):
        if self.is_visible:
            user32.SetWindowPos(
                self.hwnd,
                HWND_TOP,
                0,
                0,
                0,
                0,
                SWP_NOACTIVATE | SWP_NOOWNERZORDER | SWP_HIDEWINDOW,
            )
            self.photo_ref = None
            self.is_visible = False

    def destroy(self):
        try:
            self.hide()
            self.win.destroy()
        except Exception:
            pass


class ClickThroughBlurOverlay:
    """
    Manages a pool of small click-through patch windows (MiniPatchWindow).
    Each window only covers one specific blurred body part on the girl —
    never covering the rest of the screen.
    """

    def __init__(self, master: tk.Misc, monitor: dict, max_patches: int = 18):
        self.master = master
        self.monitor = monitor
        self.left = monitor["left"]
        self.top = monitor["top"]
        self.width = monitor["width"]
        self.height = monitor["height"]
        self.max_patches = max_patches
        self.obs_visible = False

        self.pool = [MiniPatchWindow(master) for _ in range(8)]

    def set_obs_capture_visible(self, visible: bool):
        self.obs_visible = bool(visible)
        for pw in self.pool:
            pw.set_obs_capture_visible(self.obs_visible)

    def set_monitor(self, monitor: dict):
        self.monitor = monitor
        self.left = monitor["left"]
        self.top = monitor["top"]
        self.width = monitor["width"]
        self.height = monitor["height"]
        self.clear_overlay()

    def clear_overlay(self):
        for pw in self.pool:
            pw.hide()

    def render_patches(self, patches_data):
        if not patches_data:
            self.clear_overlay()
            return

        active_patches = patches_data[: self.max_patches]

        while len(self.pool) < len(active_patches):
            pw = MiniPatchWindow(self.master)
            pw.set_obs_capture_visible(self.obs_visible)
            self.pool.append(pw)

        for idx, (x1, y1, x2, y2, rgb_patch) in enumerate(active_patches):
            w = int(x2 - x1)
            h = int(y2 - y1)
            screen_x = int(self.left + x1)
            screen_y = int(self.top + y1)
            self.pool[idx].show_patch(screen_x, screen_y, w, h, rgb_patch)

        for idx in range(len(active_patches), len(self.pool)):
            self.pool[idx].hide()

    def destroy(self):
        for pw in self.pool:
            pw.destroy()
        self.pool.clear()

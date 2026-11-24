import os
import sys
import time
import cv2
import numpy as np
import onnxruntime
import nudenet


def get_base_dir() -> str:
    """
    Resolves the base directory whether running from source (.py)
    or packaged as a standalone executable (.exe via PyInstaller).
    """
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(os.path.abspath(sys.executable))
        if os.path.exists(os.path.join(exe_dir, "models")):
            return exe_dir
        if hasattr(sys, "_MEIPASS"):
            return sys._MEIPASS
        return exe_dir
    return os.path.dirname(os.path.abspath(__file__))


NUDENET_LABELS = [
    "FEMALE_GENITALIA_COVERED",  # 0
    "FACE_FEMALE",               # 1
    "BUTTOCKS_EXPOSED",          # 2
    "FEMALE_BREAST_EXPOSED",     # 3
    "FEMALE_GENITALIA_EXPOSED",  # 4
    "MALE_BREAST_EXPOSED",       # 5
    "ANUS_EXPOSED",              # 6
    "FEET_EXPOSED",              # 7
    "BELLY_COVERED",             # 8
    "FEET_COVERED",              # 9
    "ARMPITS_COVERED",           # 10
    "ARMPITS_EXPOSED",           # 11
    "FACE_MALE",                 # 12
    "BELLY_EXPOSED",             # 13
    "MALE_GENITALIA_EXPOSED",    # 14
    "ANUS_COVERED",              # 15
    "FEMALE_BREAST_COVERED",     # 16
    "BUTTOCKS_COVERED",          # 17
]

BOORU_LABELS = [
    "head",
    "bust",
    "boob",
    "shld",
    "sideb",
    "belly",
    "nopan",
    "butt",
    "ass",
    "split",
    "sprd",
    "vsplt",
    "vsprd",
    "hip",
    "wing",
    "feral",
    "hdrago",
    "hpony",
    "hfox",
    "hrabb",
    "hcat",
    "hbear",
    "jacko",
    "jackx",
    "hhorse",
    "hbird",
]

BOORU_HEAD_CLASSES = {
    "head",
    "hcat",
    "hfox",
    "hrabb",
    "hbear",
    "hpony",
    "hdrago",
    "hhorse",
    "hbird",
}

BOORU_STRICT_FEMALE_BODY = {
    "boob",
    "sideb",
    "nopan",
    "split",
    "sprd",
    "vsplt",
    "vsprd",
}

BOORU_CONTEXT_ATTRACTIVE = {
    "bust",
    "belly",
    "butt",
    "ass",
    "hip",
}

BOORU_CONTEXT_LIMBS = {
    "shld",
}

STRICT_FEMALE_BODY_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_BREAST_COVERED",
    "FEMALE_GENITALIA_EXPOSED",
    "FEMALE_GENITALIA_COVERED",
}

STRICT_MALE_CLASSES = {
    "FACE_MALE",
    "MALE_BREAST_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
}

FEMALE_ATTRACTIVE_DIRECT = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_BREAST_COVERED",
    "FEMALE_GENITALIA_EXPOSED",
    "FEMALE_GENITALIA_COVERED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}

GENDER_NEUTRAL_ATTRACTIVE = {
    "BELLY_EXPOSED",
    "BUTTOCKS_COVERED",
}

GENDER_NEUTRAL_LIMBS = {
    "ARMPITS_EXPOSED",
    "FEET_EXPOSED",
}


def compute_iou(b1, b2) -> float:
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / float(a1 + a2 - inter)


def compute_containment(b1, b2) -> float:
    ix1 = max(b1[0], b2[0])
    iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2])
    iy2 = min(b1[3], b2[3])
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    return inter / float(min(a1, a2))


def deduplicate_tight_boxes(boxes, max_box_area: int = 0):
    if not boxes:
        return []

    rects = [
        [int(b[0]), int(b[1]), int(b[2]), int(b[3])]
        for b in boxes
        if (b[2] - b[0]) >= 6 and (b[3] - b[1]) >= 6
    ]

    merged = True
    while merged and len(rects) > 1:
        merged = False
        new_rects = []
        used = [False] * len(rects)
        for i in range(len(rects)):
            if used[i]:
                continue
            r1 = rects[i]
            for j in range(i + 1, len(rects)):
                if used[j]:
                    continue
                r2 = rects[j]
                iou = compute_iou(r1, r2)
                cont = compute_containment(r1, r2)
                if iou >= 0.48 or cont >= 0.72:
                    ux1 = min(r1[0], r2[0])
                    uy1 = min(r1[1], r2[1])
                    ux2 = max(r1[2], r2[2])
                    uy2 = max(r1[3], r2[3])
                    u_area = (ux2 - ux1) * (uy2 - uy1)
                    if max_box_area <= 0 or u_area <= max_box_area:
                        r1 = [ux1, uy1, ux2, uy2]
                        used[j] = True
                        merged = True
            new_rects.append(r1)
        rects = new_rects
    return rects


class SmoothBoxTracker:
    """
    Velocity-predictive box tracker with motion-lead padding.
    Tracks velocity (vx, vy) of each blurred body part so that during fast motion
    (TikTok dances, action games, anime fights), the blur box leads the movement
    and never lags behind the exposed part.
    """

    def __init__(self, hold_time: float = 0.28, smoothing: float = 0.68):
        self.hold_time = hold_time
        self.smoothing = smoothing
        self.tracks = []

    def clear(self):
        self.tracks.clear()

    def update(self, raw_boxes, max_box_area: int = 0, now: float = None):
        if now is None:
            now = time.perf_counter()

        active_tracks = [
            tr for tr in self.tracks if (now - tr["last_seen"]) <= self.hold_time
        ]

        used_tracks = set()
        updated_tracks = []

        for box in raw_boxes:
            bx1, by1, bx2, by2 = box
            bw = max(1, bx2 - bx1)
            bh = max(1, by2 - by1)
            bcx = (bx1 + bx2) * 0.5
            bcy = (by1 + by2) * 0.5

            best_idx = -1
            best_score = 0.0

            for idx, tr in enumerate(active_tracks):
                if idx in used_tracks:
                    continue
                tx1, ty1, tx2, ty2 = tr["box"]
                iou = compute_iou(box, tr["box"])
                tcx = (tx1 + tx2) * 0.5
                tcy = (ty1 + ty2) * 0.5
                dist_norm = (abs(bcx - tcx) / bw) + (abs(bcy - tcy) / bh)
                score = iou + (0.35 if dist_norm < 0.60 else 0.0)
                if score > best_score and (iou > 0.16 or dist_norm < 0.55):
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                used_tracks.add(best_idx)
                tr = active_tracks[best_idx]
                old_box = tr["box"]
                old_cx = (old_box[0] + old_box[2]) * 0.5
                old_cy = (old_box[1] + old_box[3]) * 0.5

                vx = bcx - old_cx
                vy = bcy - old_cy
                old_vx = tr.get("vx", 0.0)
                old_vy = tr.get("vy", 0.0)
                smooth_vx = old_vx * 0.4 + vx * 0.6
                smooth_vy = old_vy * 0.4 + vy * 0.6

                alpha = self.smoothing
                nx1 = int(old_box[0] * (1 - alpha) + bx1 * alpha)
                ny1 = int(old_box[1] * (1 - alpha) + by1 * alpha)
                nx2 = int(old_box[2] * (1 - alpha) + bx2 * alpha)
                ny2 = int(old_box[3] * (1 - alpha) + by2 * alpha)

                # Expand leading edge in direction of fast motion so fast movement never peeks out
                lead_x = int(smooth_vx * 0.45)
                lead_y = int(smooth_vy * 0.45)
                if lead_x < 0:
                    nx1 += lead_x
                else:
                    nx2 += lead_x
                if lead_y < 0:
                    ny1 += lead_y
                else:
                    ny2 += lead_y

                updated_tracks.append(
                    {
                        "box": [nx1, ny1, nx2, ny2],
                        "vx": smooth_vx,
                        "vy": smooth_vy,
                        "last_seen": now,
                    }
                )
            else:
                updated_tracks.append(
                    {
                        "box": [int(bx1), int(by1), int(bx2), int(by2)],
                        "vx": 0.0,
                        "vy": 0.0,
                        "last_seen": now,
                    }
                )

        for idx, tr in enumerate(active_tracks):
            if idx not in used_tracks:
                updated_tracks.append(tr)

        self.tracks = updated_tracks
        final_boxes = [tr["box"] for tr in self.tracks]
        return deduplicate_tight_boxes(final_boxes, max_box_area=max_box_area)


class GenderTrackMemory:
    """
    Maintains temporal EMA gender probability per tracked character head across frames,
    with automatic scene-cut / face-change detection (via 8x8 color fingerprint) so scrolling
    from a girl video to a man video on TikTok/Shorts immediately resets the gender evaluation.
    """

    def __init__(self, ttl: float = 0.9):
        self.ttl = ttl
        self.people = []

    def clear(self):
        self.people.clear()

    @staticmethod
    def _compute_fingerprint(bgr_frame: np.ndarray, box) -> np.ndarray:
        h, w = bgr_frame.shape[:2]
        x1 = max(0, min(w - 1, int(box[0])))
        y1 = max(0, min(h - 1, int(box[1])))
        x2 = max(x1 + 1, min(w, int(box[2])))
        y2 = max(y1 + 1, min(h, int(box[3])))
        crop = bgr_frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((8, 8, 3), dtype=np.float32)
        return cv2.resize(crop, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)

    def lookup_or_update(
        self, bgr_frame: np.ndarray, box, eval_fn, now: float = None
    ) -> bool:
        if now is None:
            now = time.perf_counter()

        self.people = [p for p in self.people if (now - p["last_seen"]) <= self.ttl]

        bx1, by1, bx2, by2 = box
        bw, bh = max(1, bx2 - bx1), max(1, by2 - by1)
        bcx, bcy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
        fp = self._compute_fingerprint(bgr_frame, box)

        best_p = None
        best_score = 0.0
        for p in self.people:
            px1, py1, px2, py2 = p["box"]
            iou = compute_iou(box, p["box"])
            pcx, pcy = (px1 + px2) * 0.5, (py1 + py2) * 0.5
            dist = (abs(bcx - pcx) / bw) + (abs(bcy - pcy) / bh)
            sc = iou + (0.4 if dist < 0.55 else 0.0)
            if sc > best_score and (iou > 0.20 or dist < 0.50):
                # Check visual fingerprint diff to detect scene cuts / scrolling to a new person
                fp_diff = float(np.mean(np.abs(fp - p["fp"])))
                if fp_diff <= 24.0:
                    best_score = sc
                    best_p = p

        if best_p is not None:
            best_p["box"] = list(box)
            best_p["fp"] = fp
            best_p["last_seen"] = now
            if (now - best_p["last_eval"]) >= 0.30:
                new_prob = eval_fn()
                best_p["female_prob"] = best_p["female_prob"] * 0.45 + new_prob * 0.55
                best_p["last_eval"] = now
            return best_p["female_prob"] >= 0.56

        prob = eval_fn()
        self.people.append(
            {
                "box": list(box),
                "fp": fp,
                "female_prob": prob,
                "last_eval": now,
                "last_seen": now,
            }
        )
        return prob >= 0.56


class SmartBodyBlurDetector:
    """
    Production-ready GPU/CPU Adaptive Triple-Engine Detector:
    1. 640m.onnx (High-precision 640x640 YOLOv8m) + 320n.onnx (Ultra-light fallback for low-end CPUs)
    2. booru_yolov8n.onnx (Specialized 640x640 YOLOv8 for Anime & Stylized 2D/3D Games)
    3. vit_gender.onnx + gender_googlenet.onnx with GenderTrackMemory temporal smoothing
    """

    def __init__(self):
        base_dir = get_base_dir()
        models_dir = os.path.join(base_dir, "models")
        self._ensure_models_downloaded(models_dir)

        sess_opts = onnxruntime.SessionOptions()
        sess_opts.intra_op_num_threads = max(2, min(4, (os.cpu_count() or 4) // 2))
        sess_opts.graph_optimization_level = (
            onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        available = onnxruntime.get_available_providers()
        self.is_gpu = False
        providers = ["CPUExecutionProvider"]
        if "DmlExecutionProvider" in available:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            self.is_gpu = True
            self.hardware_label = "GPU (DirectML / كرت الشاشة)"
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.is_gpu = True
            self.hardware_label = "GPU (NVIDIA CUDA)"
        else:
            self.hardware_label = "CPU (المعالج المركزي)"

        # Load both 640m (high precision) and 320n (ultra-light mode / CPU fallback)
        path_640m = os.path.join(models_dir, "640m.onnx")
        path_320n = os.path.join(os.path.dirname(nudenet.__file__), "320n.onnx")

        self.session_640m = None
        if os.path.exists(path_640m) and os.path.getsize(path_640m) > 10_000_000:
            try:
                self.session_640m = onnxruntime.InferenceSession(
                    path_640m, sess_options=sess_opts, providers=providers
                )
            except Exception:
                self.session_640m = None

        self.session_320n = onnxruntime.InferenceSession(
            path_320n, sess_options=sess_opts, providers=providers
        )

        # Default active session
        if self.session_640m is not None:
            self.session = self.session_640m
            self.primary_res = 640
        else:
            self.session = self.session_320n
            self.primary_res = 320
        self.input_name = self.session.get_inputs()[0].name

        # 2. Anime & Stylized Games Detector: booru_yolov8n.onnx (640x640)
        path_booru = os.path.join(models_dir, "booru_yolov8n.onnx")
        self.booru_session = None
        self.booru_input_name = None
        if os.path.exists(path_booru) and os.path.getsize(path_booru) > 1_000_000:
            try:
                self.booru_session = onnxruntime.InferenceSession(
                    path_booru, sess_options=sess_opts, providers=providers
                )
                self.booru_input_name = self.booru_session.get_inputs()[0].name
            except Exception:
                self.booru_session = None

        # 2b. Anime Sensitive Censor Detector: anime_censor_n.onnx (640x640)
        path_censor = os.path.join(models_dir, "anime_censor_n.onnx")
        self.censor_session = None
        self.censor_input_name = None
        if os.path.exists(path_censor) and os.path.getsize(path_censor) > 1_000_000:
            try:
                self.censor_session = onnxruntime.InferenceSession(
                    path_censor, sess_options=sess_opts, providers=providers
                )
                self.censor_input_name = self.censor_session.get_inputs()[0].name
            except Exception:
                self.censor_session = None

        # 3. Vision Transformer Gender Classifier (Anime, 3D Games, and Real Humans)
        path_vit_gender = os.path.join(models_dir, "vit_gender.onnx")
        self.vit_gender_session = None
        self.vit_gender_input = None
        if os.path.exists(path_vit_gender) and os.path.getsize(path_vit_gender) > 10_000_000:
            try:
                self.vit_gender_session = onnxruntime.InferenceSession(
                    path_vit_gender, sess_options=sess_opts, providers=providers
                )
                self.vit_gender_input = self.vit_gender_session.get_inputs()[0].name
            except Exception:
                self.vit_gender_session = None

        # 3b. FairFace Vision Transformer Gender Classifier (GTA V / 3D Games / Diverse Angles & Lighting)
        path_fairface = os.path.join(models_dir, "fairface_gender.onnx")
        self.fairface_session = None
        self.fairface_input = None
        if os.path.exists(path_fairface) and os.path.getsize(path_fairface) > 10_000_000:
            try:
                self.fairface_session = onnxruntime.InferenceSession(
                    path_fairface, sess_options=sess_opts, providers=providers
                )
                self.fairface_input = self.fairface_session.get_inputs()[0].name
            except Exception:
                self.fairface_session = None

        # 4. Secondary Real-Face CNN Gender Classifier
        path_cnn_gender = os.path.join(models_dir, "gender_googlenet.onnx")
        self.cnn_gender_session = None
        self.cnn_gender_input = None
        if os.path.exists(path_cnn_gender):
            try:
                self.cnn_gender_session = onnxruntime.InferenceSession(
                    path_cnn_gender, sess_options=sess_opts, providers=providers
                )
                self.cnn_gender_input = self.cnn_gender_session.get_inputs()[0].name
            except Exception:
                self.cnn_gender_session = None

        self.gender_memory = GenderTrackMemory(ttl=1.2)

        # Configurable flags
        self.blur_face = True
        self.blur_hair = True
        self.blur_chest_attractive = True
        self.blur_arms_legs = True
        self.blur_full_body = False
        self.enable_anime_games = True

        self.confidence_threshold = 0.30
        self.box_padding_ratio = 0.08
        self.performance_mode = "balanced"

        self.tracker = SmoothBoxTracker(hold_time=0.28, smoothing=0.65)
        self.last_inference_ms = 0.0
        self._warmup_sessions()

    @staticmethod
    def _ensure_models_downloaded(models_dir: str):
        """Auto-downloads any missing ONNX model files on first run after cloning from GitHub."""
        os.makedirs(models_dir, exist_ok=True)
        required = {
            "booru_yolov8n.onnx": (
                1_000_000,
                "https://huggingface.co/deepghs/booru_yolo/resolve/main/yolov8n_as01/model.onnx",
            ),
            "anime_censor_n.onnx": (
                1_000_000,
                "https://huggingface.co/deepghs/anime_censor_detection/resolve/main/censor_detect_v1.0_n/model.onnx",
            ),
            "gender_googlenet.onnx": (
                1_000_000,
                "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/body_analysis/age_gender/models/gender_googlenet.onnx",
            ),
            "vit_gender.onnx": (
                10_000_000,
                "https://huggingface.co/onnx-community/gender-classification-ONNX/resolve/main/onnx/model_uint8.onnx",
            ),
            "fairface_gender.onnx": (
                10_000_000,
                "https://huggingface.co/onnx-community/fairface_gender_image_detection-ONNX/resolve/main/onnx/model_uint8.onnx",
            ),
        }
        try:
            import requests

            for fname, (min_sz, url) in required.items():
                fpath = os.path.join(models_dir, fname)
                if not os.path.exists(fpath) or os.path.getsize(fpath) < min_sz:
                    print(f"Downloading missing AI model: {fname} ...")
                    r = requests.get(url, timeout=60)
                    if r.status_code == 200 and len(r.content) >= min_sz:
                        with open(fpath, "wb") as f:
                            f.write(r.content)
        except Exception:
            pass

    def _warmup_sessions(self):
        try:
            dummy_640 = np.zeros(
                (1, 3, self.primary_res, self.primary_res), dtype=np.float32
            )
            self.session.run(None, {self.input_name: dummy_640})
            if self.booru_session is not None:
                dummy_b = np.zeros((1, 3, 640, 640), dtype=np.float32)
                self.booru_session.run(None, {self.booru_input_name: dummy_b})
            if self.vit_gender_session is not None:
                dummy_224 = np.zeros((1, 3, 224, 224), dtype=np.float32)
                self.vit_gender_session.run(None, {self.vit_gender_input: dummy_224})
        except Exception:
            pass

    def set_performance_mode(self, mode: str):
        self.performance_mode = mode
        # On CPU-only machines in 'light' mode, switch primary detector to 320n for maximum FPS
        if mode == "light" and not self.is_gpu:
            self.session = self.session_320n
            self.primary_res = 320
        elif self.session_640m is not None:
            self.session = self.session_640m
            self.primary_res = 640
        else:
            self.session = self.session_320n
            self.primary_res = 320
        self.input_name = self.session.get_inputs()[0].name

    def _prepare_letterbox(self, bgr_img: np.ndarray, target_res: int = 640):
        h, w = bgr_img.shape[:2]
        tiles = []
        transforms = []

        max_s = max(w, h)
        pad = np.zeros((max_s, max_s, 3), dtype=np.uint8)
        pad[0:h, 0:w] = bgr_img
        t_full = cv2.resize(
            pad, (target_res, target_res), interpolation=cv2.INTER_LINEAR
        )
        tiles.append(t_full)
        scale_full = max_s / float(target_res)
        transforms.append((0, 0, scale_full, scale_full))

        # On widescreen monitors (GTA V / 3D games / desktop browsers), add a Center Action Square Tile
        # so medium/small characters in the center of the screen get 2.3x higher spatial resolution!
        if w > int(h * 1.35) and (
            self.performance_mode == "high"
            or (self.performance_mode == "balanced" and self.is_gpu)
        ):
            side = h
            scale_side = side / float(target_res)
            cx_start = max(0, (w - side) // 2)
            c_crop = bgr_img[0:side, cx_start : cx_start + side]
            tiles.append(
                cv2.resize(
                    c_crop, (target_res, target_res), interpolation=cv2.INTER_LINEAR
                )
            )
            transforms.append((cx_start, 0, scale_side, scale_side))

        blobs = [
            cv2.dnn.blobFromImage(
                t,
                1.0 / 255.0,
                (target_res, target_res),
                (0, 0, 0),
                swapRB=True,
                crop=False,
            )
            for t in tiles
        ]
        return np.vstack(blobs), transforms

    def _decode_yolo_outputs(
        self,
        batch_out: np.ndarray,
        transforms,
        labels_list,
        min_thresh: float,
        orig_w: int,
        orig_h: int,
        source_tag: str,
    ):
        all_boxes_xywh = []
        all_scores = []
        all_class_ids = []

        for t_idx, (off_x, off_y, sx, sy) in enumerate(transforms):
            preds = batch_out[t_idx].T
            class_scores = preds[:, 4:]
            max_scores = np.max(class_scores, axis=1)
            mask = max_scores >= min_thresh
            if not np.any(mask):
                continue

            f_preds = preds[mask]
            f_scores = max_scores[mask]
            c_ids = np.argmax(f_preds[:, 4:], axis=1)

            cx = f_preds[:, 0]
            cy = f_preds[:, 1]
            bw = f_preds[:, 2]
            bh = f_preds[:, 3]

            x1 = off_x + (cx - bw * 0.5) * sx
            y1 = off_y + (cy - bh * 0.5) * sy
            w = bw * sx
            h = bh * sy

            for i in range(len(c_ids)):
                all_boxes_xywh.append(
                    [float(x1[i]), float(y1[i]), float(w[i]), float(h[i])]
                )
                all_scores.append(float(f_scores[i]))
                all_class_ids.append(int(c_ids[i]))

        if not all_boxes_xywh:
            return []

        indices = cv2.dnn.NMSBoxes(all_boxes_xywh, all_scores, min_thresh, 0.45)
        detections = []
        if len(indices) > 0:
            for idx in np.array(indices).flatten():
                bx, by, b_w, b_h = all_boxes_xywh[idx]
                ix1 = max(0, min(orig_w, int(bx)))
                iy1 = max(0, min(orig_h, int(by)))
                ix2 = max(0, min(orig_w, int(bx + b_w)))
                iy2 = max(0, min(orig_h, int(by + b_h)))
                if ix2 > ix1 + 8 and iy2 > iy1 + 8:
                    detections.append(
                        {
                            "class": labels_list[all_class_ids[idx]],
                            "score": float(all_scores[idx]),
                            "box": [ix1, iy1, ix2, iy2],
                            "source": source_tag,
                        }
                    )
        return detections

    def _classify_gender_all_domains(
        self, bgr_frame: np.ndarray, head_box, source_tag: str, raw_cls: str
    ) -> bool:
        """
        Evaluates whether a detected head/face belongs to a Female (True) or Male (False)
        using Vision Transformer (vit_gender.onnx) + CNN (gender_googlenet.onnx) + GenderTrackMemory.
        """
        fx1, fy1, fx2, fy2 = head_box
        fw = max(1, fx2 - fx1)
        fh = max(1, fy2 - fy1)
        orig_h, orig_w = bgr_frame.shape[:2]

        def compute_female_probability() -> float:
            pad_mult = 0.08 if source_tag == "booru" else 0.22
            cx1 = max(0, int(fx1 - fw * pad_mult))
            cy1 = max(0, int(fy1 - fh * pad_mult))
            cx2 = min(orig_w, int(fx2 + fw * pad_mult))
            cy2 = min(orig_h, int(fy2 + fh * pad_mult))

            crop = bgr_frame[cy1:cy2, cx1:cx2]
            if crop.size == 0 or crop.shape[0] < 12 or crop.shape[1] < 12:
                return 0.75 if raw_cls == "FACE_FEMALE" else 0.25

            vit_f_prob = None
            fair_f_prob = None
            cnn_f_prob = None

            if self.vit_gender_session is not None or self.fairface_session is not None:
                try:
                    rgb = (
                        cv2.cvtColor(
                            cv2.resize(
                                crop, (224, 224), interpolation=cv2.INTER_LINEAR
                            ),
                            cv2.COLOR_BGR2RGB,
                        ).astype(np.float32)
                        / 255.0
                    )
                    norm = (rgb - 0.5) / 0.5
                    blob = np.transpose(norm, (2, 0, 1))[None, ...]

                    if self.vit_gender_session is not None:
                        logits = self.vit_gender_session.run(
                            None, {self.vit_gender_input: blob}
                        )[0][0]
                        exp_l = np.exp(logits - np.max(logits))
                        probs = exp_l / np.sum(exp_l)
                        vit_f_prob = float(probs[0])  # 0 = female, 1 = male

                    if self.fairface_session is not None:
                        f_logits = self.fairface_session.run(
                            None, {self.fairface_input: blob}
                        )[0][0]
                        f_exp = np.exp(f_logits - np.max(f_logits))
                        f_probs = f_exp / np.sum(f_exp)
                        fair_f_prob = float(f_probs[0])  # 0 = Female, 1 = Male
                except Exception:
                    pass

            if source_tag == "nudenet" and self.cnn_gender_session is not None:
                try:
                    c_blob = cv2.dnn.blobFromImage(
                        crop,
                        1.0,
                        (224, 224),
                        (104.0, 117.0, 123.0),
                        swapRB=False,
                        crop=False,
                    )
                    c_preds = self.cnn_gender_session.run(
                        None, {self.cnn_gender_input: c_blob}
                    )[0][0]
                    cnn_f_prob = float(c_preds[1])
                except Exception:
                    cnn_f_prob = None

            # Ensemble Decision Logic:
            if source_tag == "booru":
                # Anime & Stylized Games: vit_gender is primary (75%), fairface reinforces male rejection (25%)
                if vit_f_prob is not None and fair_f_prob is not None:
                    return vit_f_prob * 0.76 + fair_f_prob * 0.24
                elif vit_f_prob is not None:
                    return vit_f_prob
            else:
                # GTA V (قراند), 3D Realistic Games & Real Humans:
                # Combine vit_gender + fairface_gender + cnn_gender
                if vit_f_prob is not None and fair_f_prob is not None:
                    if vit_f_prob < 0.45 or fair_f_prob < 0.40:
                        return min(vit_f_prob, fair_f_prob)
                    combined_vit = vit_f_prob * 0.55 + fair_f_prob * 0.45
                    if cnn_f_prob is not None:
                        if cnn_f_prob < 0.35:
                            return min(combined_vit, cnn_f_prob)
                        return combined_vit * 0.70 + cnn_f_prob * 0.30
                    return combined_vit
                elif vit_f_prob is not None and cnn_f_prob is not None:
                    if vit_f_prob < 0.48 or cnn_f_prob < 0.40:
                        return min(vit_f_prob, cnn_f_prob)
                    return vit_f_prob * 0.60 + cnn_f_prob * 0.40
                elif vit_f_prob is not None:
                    return vit_f_prob
                elif cnn_f_prob is not None:
                    return cnn_f_prob
            return 0.75 if raw_cls == "FACE_FEMALE" else 0.25

        return self.gender_memory.lookup_or_update(
            bgr_frame, head_box, compute_female_probability
        )

    def detect_frame(self, bgr_frame: np.ndarray, use_tracker: bool = True):
        t0 = time.perf_counter()
        orig_h, orig_w = bgr_frame.shape[:2]
        max_ratio = 0.85 if self.blur_full_body else 0.78
        max_box_area = int(orig_w * orig_h * max_ratio)

        conf = max(0.24, float(self.confidence_threshold))

        batch_primary, transforms_primary = self._prepare_letterbox(
            bgr_frame, self.primary_res
        )

        # 1. Run Primary Detector (640m.onnx or 320n.onnx)
        out_primary = self.session.run(None, {self.input_name: batch_primary})[0]
        nudenet_dets = self._decode_yolo_outputs(
            out_primary,
            transforms_primary,
            NUDENET_LABELS,
            min(0.22, conf),
            orig_w,
            orig_h,
            "nudenet",
        )

        # 2. Run Anime & Stylized Games Detector (booru_yolov8n.onnx at 640x640)
        booru_dets = []
        if self.enable_anime_games and self.booru_session is not None:
            if self.primary_res == 640:
                batch_640, transforms_640 = batch_primary, transforms_primary
            else:
                batch_640, transforms_640 = self._prepare_letterbox(bgr_frame, 640)
            out_booru = self.booru_session.run(
                None, {self.booru_input_name: batch_640}
            )[0]
            booru_dets = self._decode_yolo_outputs(
                out_booru,
                transforms_640,
                BOORU_LABELS,
                max(0.26, conf),
                orig_w,
                orig_h,
                "booru",
            )

            # 2b. Run Anime Sensitive Area Censor Detector (anime_censor_n.onnx)
            if self.censor_session is not None:
                try:
                    out_censor = self.censor_session.run(
                        None, {self.censor_input_name: batch_640}
                    )[0]
                    censor_dets = self._decode_yolo_outputs(
                        out_censor,
                        transforms_640,
                        ["nipple_f", "penis", "pussy"],
                        max(0.24, conf - 0.04),
                        orig_w,
                        orig_h,
                        "booru",
                    )
                    for cd in censor_dets:
                        x1, y1, x2, y2 = cd["box"]
                        cw, ch = max(1, x2 - x1), max(1, y2 - y1)
                        if cd["class"] == "nipple_f":
                            booru_dets.append(
                                {
                                    "class": "boob",
                                    "score": cd["score"],
                                    "box": [
                                        max(0, int(x1 - cw * 0.65)),
                                        max(0, int(y1 - ch * 0.65)),
                                        min(orig_w, int(x2 + cw * 0.65)),
                                        min(orig_h, int(y2 + ch * 0.65)),
                                    ],
                                    "source": "booru",
                                }
                            )
                        elif cd["class"] == "pussy":
                            booru_dets.append(
                                {
                                    "class": "nopan",
                                    "score": cd["score"],
                                    "box": [
                                        max(0, int(x1 - cw * 0.35)),
                                        max(0, int(y1 - ch * 0.35)),
                                        min(orig_w, int(x2 + cw * 0.35)),
                                        min(orig_h, int(y2 + ch * 0.35)),
                                    ],
                                    "source": "booru",
                                }
                            )
                        elif cd["class"] == "penis":
                            nudenet_dets.append(
                                {
                                    "class": "MALE_GENITALIA_EXPOSED",
                                    "score": cd["score"],
                                    "box": cd["box"],
                                    "source": "nudenet",
                                }
                            )
                except Exception:
                    pass

        small_w = min(640, orig_w)
        small_h = max(64, int(orig_h * (small_w / float(orig_w))))
        small_bgr = cv2.resize(
            bgr_frame, (small_w, small_h), interpolation=cv2.INTER_LINEAR
        )

        raw_blur_boxes = self._fuse_and_filter_detections(
            bgr_frame, small_bgr, nudenet_dets, booru_dets, orig_w, orig_h, conf
        )

        padded_boxes = []
        pad_ratio = self.box_padding_ratio
        for x1, y1, x2, y2 in raw_blur_boxes:
            w = x2 - x1
            h = y2 - y1
            px = int(w * pad_ratio)
            py = int(h * pad_ratio)
            nx1 = max(0, x1 - px)
            ny1 = max(0, y1 - py)
            nx2 = min(orig_w, x2 + px)
            ny2 = min(orig_h, y2 + py)
            if nx2 > nx1 + 6 and ny2 > ny1 + 6:
                padded_boxes.append([nx1, ny1, nx2, ny2])

        deduped = deduplicate_tight_boxes(padded_boxes, max_box_area=max_box_area)

        if use_tracker:
            final_boxes = self.tracker.update(deduped, max_box_area=max_box_area)
        else:
            final_boxes = deduped

        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return final_boxes

    def _fuse_and_filter_detections(
        self,
        bgr_frame,
        small_bgr,
        nudenet_dets,
        booru_dets,
        orig_w,
        orig_h,
        conf_thresh,
    ):
        male_heads = []
        male_body_zones = []
        confirmed_female_heads = []
        confirmed_female_anchors = []

        def add_male_protection_zone(hbox):
            mx1, my1, mx2, my2 = hbox
            mw = max(1, mx2 - mx1)
            mh = max(1, my2 - my1)
            male_body_zones.append(
                [
                    max(0, int(mx1 - mw * 1.15)),
                    max(0, int(my1 - mh * 0.45)),
                    min(orig_w, int(mx2 + mw * 1.15)),
                    min(orig_h, int(my2 + mh * 6.0)),
                ]
            )

        for det in nudenet_dets:
            cls_name = det["class"]
            box = det["box"]
            if cls_name in ("FACE_FEMALE", "FACE_MALE"):
                if cls_name == "FACE_FEMALE" and det["score"] < conf_thresh:
                    continue
                is_female = self._classify_gender_all_domains(
                    bgr_frame, box, "nudenet", cls_name
                )
                if is_female:
                    d_copy = dict(det)
                    d_copy["class"] = "FACE_FEMALE"
                    confirmed_female_heads.append(d_copy)
                    confirmed_female_anchors.append(d_copy)
                else:
                    male_heads.append(det)
                    add_male_protection_zone(box)
            elif cls_name in ("MALE_BREAST_EXPOSED", "MALE_GENITALIA_EXPOSED"):
                mx1, my1, mx2, my2 = box
                mw, mh = max(1, mx2 - mx1), max(1, my2 - my1)
                male_body_zones.append(
                    [
                        max(0, int(mx1 - mw * 0.75)),
                        max(0, int(my1 - mh * 1.5)),
                        min(orig_w, int(mx2 + mw * 0.75)),
                        min(orig_h, int(my2 + mh * 3.0)),
                    ]
                )

        for det in booru_dets:
            cls_name = det["class"]
            box = det["box"]
            if cls_name in BOORU_HEAD_CLASSES:
                already_checked = False
                for mh in male_heads:
                    if (
                        compute_iou(box, mh["box"]) > 0.25
                        or compute_containment(box, mh["box"]) > 0.50
                    ):
                        already_checked = True
                        break
                if already_checked:
                    continue
                for fh in confirmed_female_heads:
                    if (
                        compute_iou(box, fh["box"]) > 0.25
                        or compute_containment(box, fh["box"]) > 0.50
                    ):
                        already_checked = True
                        break
                if already_checked:
                    continue

                is_female = self._classify_gender_all_domains(
                    bgr_frame, box, "booru", cls_name
                )
                if is_female:
                    d_copy = dict(det)
                    d_copy["class"] = "FACE_FEMALE"
                    confirmed_female_heads.append(d_copy)
                    confirmed_female_anchors.append(d_copy)
                else:
                    male_heads.append(det)
                    add_male_protection_zone(box)

        clean_female_heads = []
        for fh in confirmed_female_heads:
            collides = False
            for mh in male_heads:
                if (
                    compute_iou(fh["box"], mh["box"]) > 0.20
                    or compute_containment(fh["box"], mh["box"]) > 0.45
                ):
                    collides = True
                    break
            if not collides:
                clean_female_heads.append(fh)
        confirmed_female_heads = clean_female_heads
        confirmed_female_anchors = list(confirmed_female_heads)

        for det in nudenet_dets:
            if (
                det["class"] in STRICT_FEMALE_BODY_CLASSES
                and det["score"] >= conf_thresh
            ):
                if not self._is_inside_male_zone(
                    det["box"], male_body_zones, confirmed_female_heads
                ):
                    confirmed_female_anchors.append(det)

        for det in booru_dets:
            if det["class"] in BOORU_STRICT_FEMALE_BODY and det["score"] >= conf_thresh:
                if not self._is_inside_male_zone(
                    det["box"], male_body_zones, confirmed_female_heads
                ):
                    confirmed_female_anchors.append(det)

        # If no head or breast anchor was seen (e.g. mirror selfie with phone covering face, or back/lower-body shot),
        # verify high-confidence torso/lower-body detections using the Vision Transformer on the expanded person crop!
        if not confirmed_female_anchors and not male_heads:
            candidate_body_dets = [
                d
                for d in nudenet_dets
                if d["class"]
                in ("BUTTOCKS_EXPOSED", "BELLY_EXPOSED", "BUTTOCKS_COVERED")
                and d["score"] >= max(0.35, conf_thresh)
            ] + [
                d
                for d in booru_dets
                if d["class"] in ("bust", "belly", "butt", "ass", "hip")
                and d["score"] >= max(0.38, conf_thresh)
            ]
            for cdet in candidate_body_dets:
                cbox = cdet["box"]
                if not self._is_inside_male_zone(cbox, male_body_zones, []):
                    if self._verify_headless_body_is_female(
                        bgr_frame, cbox, orig_w, orig_h
                    ):
                        confirmed_female_anchors.append(cdet)

        if not confirmed_female_anchors:
            return []

        blur_boxes = []

        for f_det in confirmed_female_heads:
            fbox = f_det["box"]
            fx1, fy1, fx2, fy2 = fbox
            fw = max(1, fx2 - fx1)
            fh = max(1, fy2 - fy1)
            src = f_det.get("source", "nudenet")

            if src == "booru":
                if self.blur_face and self.blur_hair:
                    blur_boxes.append(
                        [
                            max(0, int(fx1 - fw * 0.10)),
                            max(0, int(fy1 - fh * 0.08)),
                            min(orig_w, int(fx2 + fw * 0.10)),
                            min(orig_h, int(fy2 + fh * 0.15)),
                        ]
                    )
                elif self.blur_face:
                    blur_boxes.append(
                        [
                            int(fx1 + fw * 0.15),
                            int(fy1 + fh * 0.22),
                            int(fx2 - fw * 0.15),
                            int(fy2),
                        ]
                    )
                elif self.blur_hair:
                    blur_boxes.append([fx1, fy1, fx2, int(fy1 + fh * 0.55)])
            else:
                if self.blur_face and self.blur_hair:
                    blur_boxes.append(
                        [
                            max(0, int(fx1 - fw * 0.34)),
                            max(0, int(fy1 - fh * 0.34)),
                            min(orig_w, int(fx2 + fw * 0.34)),
                            min(orig_h, int(fy2 + fh * 0.32)),
                        ]
                    )
                elif self.blur_face:
                    blur_boxes.append(list(fbox))
                elif self.blur_hair:
                    blur_boxes.append(
                        [
                            max(0, int(fx1 - fw * 0.34)),
                            max(0, int(fy1 - fh * 0.36)),
                            min(orig_w, int(fx2 + fw * 0.34)),
                            min(orig_h, int(fy1 + fh * 0.28)),
                        ]
                    )

            if self.blur_chest_attractive:
                cleavage_box = self._detect_tight_cleavage_skin(
                    small_bgr, fbox, orig_w, orig_h
                )
                if cleavage_box is not None:
                    blur_boxes.append(cleavage_box)

        for det in nudenet_dets:
            if det["score"] < conf_thresh:
                continue
            cls_name = det["class"]
            box = det["box"]

            if cls_name in FEMALE_ATTRACTIVE_DIRECT:
                if self.blur_chest_attractive and not self._is_inside_male_zone(
                    box, male_body_zones, confirmed_female_anchors
                ):
                    blur_boxes.append(list(box))

            elif cls_name in GENDER_NEUTRAL_ATTRACTIVE:
                if (
                    self.blur_chest_attractive
                    and self._belongs_to_female(box, confirmed_female_anchors)
                    and not self._is_inside_male_zone(
                        box, male_body_zones, confirmed_female_anchors
                    )
                ):
                    blur_boxes.append(list(box))

            elif cls_name in GENDER_NEUTRAL_LIMBS:
                if (
                    self.blur_arms_legs
                    and self._belongs_to_female(box, confirmed_female_anchors)
                    and not self._is_inside_male_zone(
                        box, male_body_zones, confirmed_female_anchors
                    )
                ):
                    blur_boxes.append(list(box))

        for det in booru_dets:
            cls_name = det["class"]
            box = det["box"]

            if cls_name in BOORU_STRICT_FEMALE_BODY:
                if self.blur_chest_attractive and not self._is_inside_male_zone(
                    box, male_body_zones, confirmed_female_anchors
                ):
                    blur_boxes.append(list(box))

            elif cls_name in BOORU_CONTEXT_ATTRACTIVE:
                if (
                    self.blur_chest_attractive
                    and self._belongs_to_female(box, confirmed_female_anchors)
                    and not self._is_inside_male_zone(
                        box, male_body_zones, confirmed_female_anchors
                    )
                ):
                    blur_boxes.append(list(box))

            elif cls_name in BOORU_CONTEXT_LIMBS:
                if (
                    self.blur_arms_legs
                    and self._belongs_to_female(box, confirmed_female_anchors)
                    and not self._is_inside_male_zone(
                        box, male_body_zones, confirmed_female_anchors
                    )
                ):
                    blur_boxes.append(list(box))

        if self.blur_arms_legs:
            limb_boxes = self._detect_tight_female_limbs(
                small_bgr,
                confirmed_female_heads,
                male_body_zones,
                orig_w,
                orig_h,
            )
            blur_boxes.extend(limb_boxes)

        if self.blur_full_body:
            for anchor in confirmed_female_anchors:
                fb = self._estimate_tight_female_silhouette(anchor, orig_w, orig_h)
                blur_boxes.append(fb)

        return blur_boxes

    @staticmethod
    def _is_inside_male_zone(box, male_zones, female_anchors) -> bool:
        if not male_zones:
            return False
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5

        nearest_male_dx = None
        for mz in male_zones:
            if mz[0] <= cx <= mz[2] and mz[1] <= cy <= mz[3]:
                mcx = (mz[0] + mz[2]) * 0.5
                dx = abs(cx - mcx)
                if nearest_male_dx is None or dx < nearest_male_dx:
                    nearest_male_dx = dx

        if nearest_male_dx is None:
            return False

        # Voronoi side-by-side separation:
        # If a confirmed female anchor is horizontally closer to this part than the male centerline,
        # it belongs to the girl standing next to him!
        for fa in female_anchors:
            fb = fa["box"]
            if compute_iou(box, fb) > 0.20:
                return False
            fcx = (fb[0] + fb[2]) * 0.5
            f_dx = abs(cx - fcx)
            if f_dx < nearest_male_dx * 0.92:
                return False
        return True

    @staticmethod
    def _belongs_to_female(part_box, female_anchors) -> bool:
        px1, py1, px2, py2 = part_box
        pcx = (px1 + px2) * 0.5
        pcy = (py1 + py2) * 0.5

        for anchor in female_anchors:
            ax1, ay1, ax2, ay2 = anchor["box"]
            aw = max(1, ax2 - ax1)
            ah = max(1, ay2 - ay1)
            acx = (ax1 + ax2) * 0.5
            acy = (ay1 + ay2) * 0.5

            if anchor["class"] == "FACE_FEMALE":
                if (
                    abs(pcx - acx) <= aw * 2.0
                    and (ay1 - ah * 0.3) <= pcy <= (ay2 + ah * 6.5)
                ):
                    return True
            else:
                if abs(pcx - acx) <= aw * 1.8 and abs(pcy - acy) <= ah * 3.5:
                    return True
        return False

    def _verify_headless_body_is_female(
        self, bgr_frame: np.ndarray, part_box, orig_w: int, orig_h: int
    ) -> bool:
        """
        When a person's face is off-screen or covered by a phone (mirror selfie / back shot),
        uses the Vision Transformer (vit_gender.onnx) on the expanded person silhouette crop
        to verify whether the person is Female (>= 0.62) vs Male.
        """
        if self.vit_gender_session is None:
            return False
        bx1, by1, bx2, by2 = part_box
        bw = max(1, bx2 - x1) if (x1 := bx1) is not None else 1
        bh = max(1, by2 - by1)
        cx1 = max(0, int(bx1 - bw * 0.55))
        cy1 = max(0, int(by1 - bh * 1.15))
        cx2 = min(orig_w, int(bx2 + bw * 0.55))
        cy2 = min(orig_h, int(by2 + bh * 0.85))

        crop = bgr_frame[cy1:cy2, cx1:cx2]
        if crop.size == 0 or crop.shape[0] < 24 or crop.shape[1] < 24:
            return False

        def eval_body():
            try:
                rgb = (
                    cv2.cvtColor(
                        cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR),
                        cv2.COLOR_BGR2RGB,
                    ).astype(np.float32)
                    / 255.0
                )
                norm = (rgb - 0.5) / 0.5
                blob = np.transpose(norm, (2, 0, 1))[None, ...]
                logits = self.vit_gender_session.run(
                    None, {self.vit_gender_input: blob}
                )[0][0]
                exp_l = np.exp(logits - np.max(logits))
                probs = exp_l / np.sum(exp_l)
                return float(probs[0])
            except Exception:
                return 0.0

        f_prob = eval_body()
        return f_prob >= 0.62

    def _detect_tight_cleavage_skin(self, small_bgr, face_box, orig_w, orig_h):
        sh, sw = small_bgr.shape[:2]
        sx = sw / float(orig_w)
        sy = sh / float(orig_h)

        fx1, fy1, fx2, fy2 = face_box
        fw = max(1, fx2 - fx1)
        fh = max(1, fy2 - fy1)

        zx1 = max(0, int((fx1 - fw * 0.22) * sx))
        zy1 = max(0, int((fy2 + fh * 0.03) * sy))
        zx2 = min(sw, int((fx2 + fw * 0.22) * sx))
        zy2 = min(sh, int((fy2 + fh * 1.32) * sy))

        if zx2 <= zx1 + 3 or zy2 <= zy1 + 3:
            return None

        roi = small_bgr[zy1:zy2, zx1:zx2]
        ref_face = small_bgr[
            max(0, int((fy1 + fh * 0.2) * sy)) : min(sh, int((fy2 - fh * 0.2) * sy)),
            max(0, int((fx1 + fw * 0.2) * sx)) : min(sw, int((fx2 - fw * 0.2) * sx)),
        ]
        skin_mask = self._adaptive_skin_mask(roi, ref_face)
        skin_ratio = np.count_nonzero(skin_mask) / float(skin_mask.size)
        if skin_ratio < 0.22:
            return None

        ys, xs = np.where(skin_mask > 0)
        if len(xs) < 6:
            return None

        bx1 = int((zx1 + int(np.min(xs))) / sx)
        by1 = int((zy1 + int(np.min(ys))) / sy)
        bx2 = int((zx1 + int(np.max(xs)) + 1) / sx)
        by2 = int((zy1 + int(np.max(ys)) + 1) / sy)
        if bx2 > bx1 + 8 and by2 > by1 + 8:
            return [bx1, by1, bx2, by2]
        return None

    def _detect_tight_female_limbs(
        self,
        small_bgr,
        female_faces,
        male_body_zones,
        orig_w,
        orig_h,
    ):
        sh, sw = small_bgr.shape[:2]
        sx = sw / float(orig_w)
        sy = sh / float(orig_h)
        limb_boxes = []

        for f_det in female_faces:
            fx1, fy1, fx2, fy2 = f_det["box"]
            fw = max(1, fx2 - fx1)
            fh = max(1, fy2 - fy1)

            zx1 = max(0, int((fx1 - fw * 1.25) * sx))
            zy1 = max(0, int((fy2 + fh * 0.12) * sy))
            zx2 = min(sw, int((fx2 + fw * 1.25) * sx))
            zy2 = min(sh, int((fy2 + fh * 5.0) * sy))

            if zx2 <= zx1 + 6 or zy2 <= zy1 + 10:
                continue

            roi = small_bgr[zy1:zy2, zx1:zx2]
            ref_face = small_bgr[
                max(0, int((fy1 + fh * 0.2) * sy)) : min(sh, int((fy2 - fh * 0.2) * sy)),
                max(0, int((fx1 + fw * 0.2) * sx)) : min(sw, int((fx2 - fw * 0.2) * sx)),
            ]
            skin_mask = self._adaptive_skin_mask(roi, ref_face)

            fcx_orig = (fx1 + fx2) * 0.5
            for mz in male_body_zones:
                mcx_orig = (mz[0] + mz[2]) * 0.5
                mid_x_orig = (fcx_orig + mcx_orig) * 0.5
                # Clip male exclusion zone at the Voronoi midpoint between the man and the girl
                eff_mz0 = mz[0] if mcx_orig < fcx_orig else max(mz[0], int(mid_x_orig))
                eff_mz2 = min(mz[2], int(mid_x_orig)) if mcx_orig < fcx_orig else mz[2]
                mx1 = max(0, int(eff_mz0 * sx) - zx1)
                my1 = max(0, int(mz[1] * sy) - zy1)
                mx2 = min(zx2 - zx1, int(eff_mz2 * sx) - zx1)
                my2 = min(zy2 - zy1, int(mz[3] * sy) - zy1)
                if mx2 > mx1 and my2 > my1:
                    skin_mask[my1:my2, mx1:mx2] = 0

            total_skin_ratio = np.count_nonzero(skin_mask) / float(skin_mask.size)
            if total_skin_ratio > 0.52 or total_skin_ratio < 0.03:
                continue

            roi_h, roi_w = skin_mask.shape[:2]
            num_bands = 2
            band_h = max(8, roi_h // num_bands)
            max_limb_w = max(12, int(fw * 1.20 * sx))
            min_limb_area = max(80, int((fw * sx) * (fh * sy) * 0.065))

            for b_idx in range(num_bands):
                by_start = b_idx * band_h
                by_end = roi_h if b_idx == num_bands - 1 else (b_idx + 1) * band_h
                band_mask = skin_mask[by_start:by_end, :]
                if band_mask.size == 0:
                    continue

                num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                    band_mask, connectivity=8
                )
                for lbl in range(1, num_labels):
                    area = stats[lbl, cv2.CC_STAT_AREA]
                    cw = stats[lbl, cv2.CC_STAT_WIDTH]
                    ch = stats[lbl, cv2.CC_STAT_HEIGHT]
                    if area < min_limb_area or cw > max_limb_w:
                        continue
                    fill_ratio = area / float(max(1, cw * ch))
                    if fill_ratio < 0.34:
                        continue

                    cx = stats[lbl, cv2.CC_STAT_LEFT]
                    cy = stats[lbl, cv2.CC_STAT_TOP]
                    bx1 = int((zx1 + cx) / sx)
                    by1 = int((zy1 + by_start + cy) / sy)
                    bx2 = int((zx1 + cx + cw) / sx)
                    by2 = int((zy1 + by_start + cy + ch) / sy)
                    if bx2 > bx1 + 20 and by2 > by1 + 20:
                        limb_boxes.append([bx1, by1, bx2, by2])

        return limb_boxes

    @classmethod
    def _adaptive_skin_mask(
        cls, roi_bgr: np.ndarray, ref_face_bgr: np.ndarray = None
    ) -> np.ndarray:
        """
        Combines strict YCrCb+HSV human skin segmentation with adaptive face skin-tone
        sampling so colored streamer LED lights or TikTok filters still detect bare skin.
        """
        base_skin = cls._strict_skin_mask(roi_bgr)
        if ref_face_bgr is None or ref_face_bgr.size == 0 or ref_face_bgr.shape[0] < 4 or ref_face_bgr.shape[1] < 4:
            return base_skin

        try:
            ref_ycrcb = cv2.cvtColor(ref_face_bgr, cv2.COLOR_BGR2YCrCb)
            med_y = int(np.median(ref_ycrcb[:, :, 0]))
            med_cr = int(np.median(ref_ycrcb[:, :, 1]))
            med_cb = int(np.median(ref_ycrcb[:, :, 2]))
            if 45 <= med_y <= 240 and 126 <= med_cr <= 182 and 75 <= med_cb <= 135:
                roi_ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
                low = np.array(
                    [max(40, med_y - 52), max(126, med_cr - 12), max(74, med_cb - 12)],
                    dtype=np.uint8,
                )
                high = np.array(
                    [min(250, med_y + 52), min(182, med_cr + 12), min(135, med_cb + 12)],
                    dtype=np.uint8,
                )
                adapt = cv2.inRange(roi_ycrcb, low, high)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                adapt = cv2.morphologyEx(adapt, cv2.MORPH_OPEN, kernel)
                return cv2.bitwise_or(base_skin, adapt)
        except Exception:
            pass
        return base_skin

    @staticmethod
    def _strict_skin_mask(roi_bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        mask_ycrcb = cv2.inRange(
            ycrcb,
            np.array([50, 135, 80], dtype=np.uint8),
            np.array([245, 173, 125], dtype=np.uint8),
        )
        mask_hsv = cv2.inRange(
            hsv,
            np.array([0, 22, 60], dtype=np.uint8),
            np.array([24, 175, 252], dtype=np.uint8),
        )
        skin = cv2.bitwise_and(mask_ycrcb, mask_hsv)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, kernel)
        return skin

    @staticmethod
    def _estimate_tight_female_silhouette(anchor, orig_w, orig_h):
        cls_name = anchor["class"]
        x1, y1, x2, y2 = anchor["box"]
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        if cls_name == "FACE_FEMALE":
            return [
                max(0, int(x1 - w * 0.85)),
                max(0, int(y1 - h * 0.30)),
                min(orig_w, int(x2 + w * 0.85)),
                min(orig_h, int(y2 + h * 4.50)),
            ]
        return [
            max(0, int(x1 - w * 0.35)),
            max(0, int(y1 - h * 0.60)),
            min(orig_w, int(x2 + w * 0.35)),
            min(orig_h, int(y2 + h * 2.00)),
        ]


_FEATHER_MASK_CACHE = {}


def _get_feathered_alpha_mask(w: int, h: int) -> np.ndarray:
    """
    Creates a smooth feathered rounded-capsule alpha mask (32x32 resized to h x w)
    so the outer 12% edges of the blur patch blend smoothly into the underlying scene
    without harsh rectangular corners.
    """
    key = (w // 8, h // 8)
    mask32 = _FEATHER_MASK_CACHE.get(key)
    if mask32 is None:
        m = np.zeros((32, 32), dtype=np.float32)
        cv2.rectangle(m, (2, 2), (29, 29), 1.0, -1)
        m = cv2.GaussianBlur(m, (7, 7), 0)
        m = np.clip(m * 1.18, 0.0, 1.0)
        _FEATHER_MASK_CACHE[key] = m
        mask32 = m
    full_m = cv2.resize(mask32, (w, h), interpolation=cv2.INTER_LINEAR)
    return full_m[:, :, None]


def apply_blur_to_patch(
    bgr_patch: np.ndarray,
    mode: str = "gaussian",
    intensity: int = 85,
    feather: bool = True,
) -> np.ndarray:
    h, w = bgr_patch.shape[:2]
    if h <= 4 or w <= 4:
        return bgr_patch

    if mode == "solid":
        out = np.full_like(bgr_patch, (28, 25, 34))
        cv2.rectangle(out, (0, 0), (w - 1, h - 1), (82, 72, 96), 2)
        return out

    factor = max(6, int(6 + (intensity / 100.0) * 20))
    small_w = max(2, w // factor)
    small_h = max(2, h // factor)

    small = cv2.resize(bgr_patch, (small_w, small_h), interpolation=cv2.INTER_AREA)

    if mode == "pixelate":
        blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        small = cv2.GaussianBlur(small, (5, 5), 0)
        small = cv2.GaussianBlur(small, (5, 5), 0)
        blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

    if feather and w >= 24 and h >= 24 and w * h < 900_000:
        alpha = _get_feathered_alpha_mask(w, h)
        blended = bgr_patch.astype(np.float32) * (1.0 - alpha) + blurred.astype(
            np.float32
        ) * alpha
        return blended.astype(np.uint8)
    return blurred

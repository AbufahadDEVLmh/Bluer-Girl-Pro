import base64
from collections import OrderedDict
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import cv2
import numpy as np

from detector import SmartBodyBlurDetector, compute_iou, deduplicate_tight_boxes


class VideoPersistenceTracker:
    """
    Maintains zero-flicker blur persistence for browser videos captured via captureVisibleTab
    (such as TikTok, YouTube Shorts, Instagram Reels, Twitch, Kick).
    Uses outer-ring context fingerprinting around active blur patches so that even when
    captureVisibleTab captures the blur patch itself, the patch stays locked onto the girl
    until she moves or the video scene cuts.
    """

    def __init__(self):
        self.videos = {}

    def clear_video(self, video_id: str):
        if video_id in self.videos:
            del self.videos[video_id]

    @staticmethod
    def _scene_fingerprint(bgr: np.ndarray, mask_boxes) -> np.ndarray:
        h, w = bgr.shape[:2]
        small = cv2.resize(bgr, (24, 24), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        valid_mask = np.ones((24, 24), dtype=bool)
        for bx1, by1, bx2, by2 in mask_boxes:
            sx1 = max(0, min(24, int((bx1 / float(w)) * 24)))
            sy1 = max(0, min(24, int((by1 / float(h)) * 24)))
            sx2 = max(0, min(24, int(np.ceil((bx2 / float(w)) * 24))))
            sy2 = max(0, min(24, int(np.ceil((by2 / float(h)) * 24))))
            valid_mask[sy1:sy2, sx1:sx2] = False
        return small, valid_mask

    @staticmethod
    def _outer_ring_fingerprint(bgr: np.ndarray, box) -> np.ndarray:
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = box
        bw = max(4, x2 - x1)
        bh = max(4, y2 - y1)
        pad_x = max(6, int(bw * 0.25))
        pad_y = max(6, int(bh * 0.25))

        ox1 = max(0, x1 - pad_x)
        oy1 = max(0, y1 - pad_y)
        ox2 = min(w, x2 + pad_x)
        oy2 = min(h, y2 + pad_y)

        outer = bgr[oy1:oy2, ox1:ox2]
        if outer.size == 0:
            return np.zeros((12, 12, 3), dtype=np.float32)

        resized = cv2.resize(outer, (12, 12), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        resized[3:9, 3:9, :] = 0.0
        return resized

    def update(self, video_id: str, bgr: np.ndarray, detected_boxes):
        now = time.perf_counter()
        h, w = bgr.shape[:2]

        if len(self.videos) > 35:
            stale = [
                k
                for k, v in self.videos.items()
                if (now - v.get("last_update", 0)) > 15.0
            ]
            for k in stale:
                del self.videos[k]

        state = self.videos.get(video_id)
        if state is None:
            tracks = []
            for b in detected_boxes:
                tracks.append(
                    {
                        "box": list(b),
                        "ring_fp": self._outer_ring_fingerprint(bgr, b),
                        "ts": now,
                        "last_direct": now,
                    }
                )
            scene_fp, _ = self._scene_fingerprint(bgr, detected_boxes)
            self.videos[video_id] = {
                "tracks": tracks,
                "scene_fp": scene_fp,
                "last_update": now,
            }
            return detected_boxes

        old_tracks = state["tracks"]
        old_boxes = [t["box"] for t in old_tracks]

        cur_scene_fp, valid_mask = self._scene_fingerprint(bgr, old_boxes)
        prev_scene_fp = state["scene_fp"]
        if np.any(valid_mask):
            scene_diff = float(
                np.mean(np.abs(cur_scene_fp[valid_mask] - prev_scene_fp[valid_mask]))
            )
        else:
            scene_diff = 0.0

        if scene_diff > 24.0:
            old_tracks = []

        new_tracks = []
        used_old = set()

        for b in detected_boxes:
            best_idx = -1
            best_iou = 0.0
            for idx, tr in enumerate(old_tracks):
                if idx in used_old:
                    continue
                iou = compute_iou(b, tr["box"])
                if iou > best_iou and iou > 0.15:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0:
                used_old.add(best_idx)
                ob = old_tracks[best_idx]["box"]
                alpha = 0.72
                sb = [
                    int(ob[0] * (1 - alpha) + b[0] * alpha),
                    int(ob[1] * (1 - alpha) + b[1] * alpha),
                    int(ob[2] * (1 - alpha) + b[2] * alpha),
                    int(ob[3] * (1 - alpha) + b[3] * alpha),
                ]
            else:
                sb = list(b)

            new_tracks.append(
                {
                    "box": sb,
                    "ring_fp": self._outer_ring_fingerprint(bgr, sb),
                    "ts": now,
                    "last_direct": now,
                }
            )

        for idx, tr in enumerate(old_tracks):
            if idx in used_old:
                continue
            age = now - tr["last_direct"]
            if age > 4.0:
                continue

            cur_ring = self._outer_ring_fingerprint(bgr, tr["box"])
            ring_diff = float(
                np.mean(np.abs(cur_ring - tr["ring_fp"])) * (144.0 / 108.0)
            )
            if ring_diff <= 19.5 or age <= 0.32:
                tr["ts"] = now
                new_tracks.append(tr)

        final_boxes = deduplicate_tight_boxes(
            [t["box"] for t in new_tracks], max_box_area=int(w * h * 0.82)
        )
        updated_scene_fp, _ = self._scene_fingerprint(bgr, final_boxes)
        self.videos[video_id] = {
            "tracks": new_tracks,
            "scene_fp": updated_scene_fp,
            "last_update": now,
        }
        return final_boxes


class ChromeAIBridgeServer:
    """
    High-Speed HTTP/1.1 Keep-Alive Local AI Server (http://127.0.0.1:17890)
    with LRU Image Hash Caching and Live Telemetry for the Bluer Girl Chrome Extension.
    """

    def __init__(self, detector: SmartBodyBlurDetector = None, port: int = 17890):
        self.detector = detector if detector is not None else SmartBodyBlurDetector()
        self.port = port
        self.video_tracker = VideoPersistenceTracker()
        self._lock = threading.Lock()
        self._httpd = None
        self._thread = None

        # LRU cache for static images/thumbnails: {hash_key: norm_boxes}
        self._image_lru = OrderedDict()
        self._max_lru = 600

        # Live Telemetry
        self.total_requests = 0
        self.cache_hits = 0
        self.total_blurred_detections = 0
        self.avg_ms = 0.0

    def _settings_signature(self, settings: dict) -> str:
        if not settings:
            return "default"
        return (
            f"{settings.get('blur_face',True)}_{settings.get('blur_hair',True)}_"
            f"{settings.get('blur_chest',True)}_{settings.get('blur_limbs',True)}_"
            f"{settings.get('enable_anime_games',True)}_{settings.get('blur_fullbody',False)}_"
            f"{settings.get('padding_pct',10)}_{settings.get('sens_pct',74)}"
        )

    def detect_base64_image(
        self,
        data_url_or_b64: str,
        settings: dict = None,
        crop: dict = None,
        video_id: str = None,
    ):
        self.total_requests += 1
        if "," in data_url_or_b64:
            data_url_or_b64 = data_url_or_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(data_url_or_b64)

        # Check LRU cache for static images (when not a live video frame)
        cache_key = None
        if not video_id and not crop:
            sig = self._settings_signature(settings)
            digest = hashlib.md5(img_bytes).hexdigest()
            cache_key = f"{digest}_{sig}"
            with self._lock:
                if cache_key in self._image_lru:
                    self._image_lru.move_to_end(cache_key)
                    self.cache_hits += 1
                    return self._image_lru[cache_key], 0.1

        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            return [], 0.0

        if crop:
            ih, iw = bgr.shape[:2]
            vw = max(1.0, float(crop.get("vw", iw)))
            vh = max(1.0, float(crop.get("vh", ih)))
            sx = iw / vw
            sy = ih / vh
            cx1 = max(0, min(iw - 2, int(float(crop.get("x", 0)) * sx)))
            cy1 = max(0, min(ih - 2, int(float(crop.get("y", 0)) * sy)))
            cx2 = max(
                cx1 + 2,
                min(iw, int((float(crop.get("x", 0)) + float(crop.get("w", vw))) * sx)),
            )
            cy2 = max(
                cy1 + 2,
                min(ih, int((float(crop.get("y", 0)) + float(crop.get("h", vh))) * sy)),
            )
            bgr = bgr[cy1:cy2, cx1:cx2]
            if bgr.size == 0 or bgr.shape[0] < 16 or bgr.shape[1] < 16:
                return [], 0.0

        h, w = bgr.shape[:2]

        with self._lock:
            if settings:
                self.detector.blur_face = bool(settings.get("blur_face", True))
                self.detector.blur_hair = bool(settings.get("blur_hair", True))
                self.detector.blur_chest_attractive = bool(
                    settings.get("blur_chest", True)
                )
                self.detector.blur_arms_legs = bool(settings.get("blur_limbs", True))
                self.detector.blur_full_body = bool(
                    settings.get("blur_fullbody", False)
                )
                self.detector.enable_anime_games = bool(
                    settings.get("enable_anime_games", True)
                )
                if "padding_pct" in settings:
                    self.detector.box_padding_ratio = (
                        float(settings["padding_pct"]) / 100.0
                    )
                if "sens_pct" in settings:
                    sens = float(settings["sens_pct"])
                    thresh = 0.44 - ((sens - 50.0) / 40.0) * 0.20
                    self.detector.confidence_threshold = round(
                        max(0.24, min(0.45, thresh)), 2
                    )

            raw_boxes = self.detector.detect_frame(bgr, use_tracker=False)
            ms = self.detector.last_inference_ms
            self.avg_ms = (self.avg_ms * 0.85 + ms * 0.15) if self.avg_ms > 0 else ms

            if video_id:
                boxes = self.video_tracker.update(video_id, bgr, raw_boxes)
            else:
                boxes = raw_boxes

            if boxes:
                self.total_blurred_detections += len(boxes)

            norm_boxes = [
                [
                    round(max(0.0, min(1.0, b[0] / float(w))), 4),
                    round(max(0.0, min(1.0, b[1] / float(h))), 4),
                    round(max(0.0, min(1.0, b[2] / float(w))), 4),
                    round(max(0.0, min(1.0, b[3] / float(h))), 4),
                ]
                for b in boxes
            ]

            if cache_key is not None:
                self._image_lru[cache_key] = norm_boxes
                if len(self._image_lru) > self._max_lru:
                    self._image_lru.popitem(last=False)

        return norm_boxes, round(ms, 1)

    def start_in_background(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send_json(self, status_code: int, payload: dict):
                raw = json.dumps(payload).encode("utf-8")
                self.send_response(status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Connection", "keep-alive")
                self.end_headers()
                self.wfile.write(raw)

            def do_OPTIONS(self):
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

            def do_GET(self):
                if self.path.startswith("/health") or self.path.startswith("/stats"):
                    self._send_json(
                        200,
                        {
                            "status": "ok",
                            "hardware": bridge.detector.hardware_label,
                            "gpu": bridge.detector.is_gpu,
                            "total_requests": bridge.total_requests,
                            "cache_hits": bridge.cache_hits,
                            "total_blurred": bridge.total_blurred_detections,
                            "avg_ms": round(bridge.avg_ms, 1),
                        },
                    )
                else:
                    self._send_json(404, {"error": "not found"})

            def do_POST(self):
                if self.path.startswith("/detect"):
                    try:
                        length = int(self.headers.get("Content-Length", 0))
                        body = self.rfile.read(length)
                        data = json.loads(body.decode("utf-8"))
                        if data.get("reset_video") and data.get("video_id"):
                            with bridge._lock:
                                bridge.video_tracker.clear_video(data["video_id"])
                            self._send_json(200, {"ok": True, "boxes": []})
                            return

                        img_b64 = data.get("image", "")
                        settings = data.get("settings", None)
                        crop = data.get("crop", None)
                        video_id = data.get("video_id", None)
                        norm_boxes, ms = bridge.detect_base64_image(
                            img_b64, settings, crop=crop, video_id=video_id
                        )
                        self._send_json(
                            200,
                            {
                                "ok": True,
                                "boxes": norm_boxes,
                                "ms": ms,
                                "hardware": bridge.detector.hardware_label,
                            },
                        )
                    except Exception as e:
                        self._send_json(
                            500, {"ok": False, "error": str(e), "boxes": []}
                        )
                else:
                    self._send_json(404, {"error": "not found"})

            def log_message(self, format, *args):
                pass

        try:
            self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
            self._thread = threading.Thread(
                target=self._httpd.serve_forever, daemon=True
            )
            self._thread.start()
            return True
        except Exception:
            return False

    def stop(self):
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
            self._httpd = None


if __name__ == "__main__":
    print(
        "Starting Bluer Girl Pro v5.0 Chrome AI Server (HTTP/1.1 Keep-Alive) on http://127.0.0.1:17890 ..."
    )
    server = ChromeAIBridgeServer()
    if server.start_in_background():
        print(f"Server active! Hardware: {server.detector.hardware_label}")
        try:
            while True:
                threading.Event().wait(3600)
        except KeyboardInterrupt:
            server.stop()
    else:
        print("Server is already running on port 17890.")

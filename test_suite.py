import base64
import json
import os
import time
import zipfile
import cv2
import numpy as np
import requests

from detector import SmartBodyBlurDetector
from chrome_ai_server import ChromeAIBridgeServer
from video_processor import process_video_file


def run_all_tests():
    print("=" * 60)
    print("🚀 Running Bluer Girl Pro v5.0 Full Verification & Benchmark Suite")
    print("=" * 60)

    # 1. Verify Detector & Models
    det = SmartBodyBlurDetector()
    print(f"[1/5] Hardware Acceleration: {det.hardware_label} (is_gpu={det.is_gpu})")
    assert det.session_640m is not None, "640m.onnx failed to load"
    assert det.booru_session is not None, "booru_yolov8n.onnx failed to load"
    assert det.censor_session is not None, "anime_censor_n.onnx failed to load"
    assert det.vit_gender_session is not None, "vit_gender.onnx failed to load"
    assert det.fairface_session is not None, "fairface_gender.onnx failed to load"
    assert det.cnn_gender_session is not None, "gender_googlenet.onnx failed to load"
    print("      ✅ All 6 Neural Network ONNX sessions loaded successfully!")

    # 2. Test HTTP/1.1 Server & LRU Cache
    srv = ChromeAIBridgeServer(detector=det, port=17895)
    srv.start_in_background()
    try:
        f_bytes = requests.get(
            "https://huggingface.co/SomeDutchGuy/animeGender2/resolve/main/images/anime_female.jpg",
            timeout=15,
        ).content
        b64_f = base64.b64encode(f_bytes).decode("ascii")

        t0 = time.perf_counter()
        r1 = requests.post(
            "http://127.0.0.1:17895/detect", json={"image": b64_f}, timeout=10
        ).json()
        t_first = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        r2 = requests.post(
            "http://127.0.0.1:17895/detect", json={"image": b64_f}, timeout=10
        ).json()
        t_cached = (time.perf_counter() - t0) * 1000.0

        print(
            f"[2/5] HTTP/1.1 Keep-Alive & LRU Cache: First={t_first:.1f}ms, Cached={t_cached:.1f}ms, Boxes={r1['boxes']}"
        )
        assert len(r1["boxes"]) > 0, "Anime female was not blurred!"
        assert r2["ms"] <= 0.5, "LRU cache did not hit!"
    finally:
        srv.stop()

    # 3. Test Video File Processor on synthetic 12-frame MP4 clip
    f_img = cv2.imdecode(np.frombuffer(f_bytes, np.uint8), cv2.IMREAD_COLOR)
    test_in_mp4 = "test_input_temp.mp4"
    test_out_mp4 = "test_output_temp.mp4"
    h, w = f_img.shape[:2]
    vw = cv2.VideoWriter(test_in_mp4, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h))
    for _ in range(10):
        vw.write(f_img)
    vw.release()

    res_vid = process_video_file(test_in_mp4, test_out_mp4, det)
    print(
        f"[3/5] Video Processor Test: Processed {res_vid['frames_processed']} frames, Blurred {res_vid['blurred_frames']} frames in {res_vid['seconds']}s"
    )
    assert res_vid["blurred_frames"] == 10 and os.path.exists(test_out_mp4)
    os.remove(test_in_mp4)
    os.remove(test_out_mp4)

    # 4. Package Chrome Extension into ready-to-publish ZIP
    zip_path = "BluerGirl_Chrome_Extension.zip"
    ext_dir = "chrome_extension"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(ext_dir):
            for file in files:
                full_p = os.path.join(root, file)
                rel_p = os.path.relpath(full_p, ext_dir)
                zf.write(full_p, rel_p)
    print(
        f"[4/5] Packaged Chrome Extension for Public Release: {zip_path} ({os.path.getsize(zip_path)} bytes)"
    )

    print("[5/5] 🎉 ALL V5.0 TESTS & PACKAGING PASSED 100%!")


if __name__ == "__main__":
    run_all_tests()

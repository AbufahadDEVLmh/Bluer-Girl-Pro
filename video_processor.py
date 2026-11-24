import os
import shutil
import subprocess
import time
import cv2
import numpy as np

from detector import SmartBodyBlurDetector, SmoothBoxTracker, apply_blur_to_patch


def process_video_file(
    input_path: str,
    output_path: str,
    detector: SmartBodyBlurDetector,
    blur_mode: str = "gaussian",
    blur_intensity: int = 88,
    progress_callback=None,
    cancel_event=None,
) -> dict:
    """
    Processes a video file (.mp4, .mkv, .mov, .avi, .webm) frame-by-frame on the GPU,
    applying smooth tracked blur strictly to detected female exposed parts,
    and remuxes original audio if ffmpeg is available.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Video file not found: {input_path}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open input video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1.0 or fps > 240.0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_video_path = output_path + ".temp_noaudio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not initialize MP4 video writer.")

    video_tracker = SmoothBoxTracker(hold_time=0.35, smoothing=0.68)
    detector.gender_memory.clear()

    frame_idx = 0
    blurred_frames_count = 0
    t_start = time.perf_counter()
    last_boxes = []

    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Run AI detection every frame (or every 2nd frame on 60fps videos for 2x speed!)
            step = 2 if fps >= 48.0 else 1
            if frame_idx % step == 0:
                raw_boxes = detector.detect_frame(frame, use_tracker=False)
                last_boxes = video_tracker.update(
                    raw_boxes, max_box_area=int(width * height * 0.85)
                )

            if last_boxes:
                blurred_frames_count += 1
                for x1, y1, x2, y2 in last_boxes:
                    cx1 = max(0, min(width - 2, int(x1)))
                    cy1 = max(0, min(height - 2, int(y1)))
                    cx2 = max(cx1 + 2, min(width, int(x2)))
                    cy2 = max(cy1 + 2, min(height, int(y2)))
                    patch = frame[cy1:cy2, cx1:cx2]
                    if patch.size > 0:
                        frame[cy1:cy2, cx1:cx2] = apply_blur_to_patch(
                            patch, mode=blur_mode, intensity=blur_intensity
                        )

            writer.write(frame)
            frame_idx += 1

            if progress_callback and (frame_idx % 5 == 0 or frame_idx == total_frames):
                elapsed = max(0.01, time.perf_counter() - t_start)
                proc_fps = frame_idx / elapsed
                pct = (
                    min(100.0, (frame_idx / float(total_frames)) * 100.0)
                    if total_frames > 0
                    else 0.0
                )
                progress_callback(frame_idx, total_frames, pct, proc_fps)
    finally:
        cap.release()
        writer.release()

    if cancel_event is not None and cancel_event.is_set():
        if os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
            except Exception:
                pass
        return {"cancelled": True}

    # Try remuxing original audio track using ffmpeg if available
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-i",
                temp_video_path,
                "-i",
                input_path,
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-map",
                "0:v:0",
                "-map",
                "1:a:0?",
                "-shortest",
                output_path,
            ]
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300,
                check=True,
            )
            if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                os.remove(temp_video_path)
            else:
                shutil.move(temp_video_path, output_path)
        except Exception:
            if os.path.exists(temp_video_path):
                shutil.move(temp_video_path, output_path)
    else:
        if os.path.exists(output_path):
            os.remove(output_path)
        shutil.move(temp_video_path, output_path)

    total_time = round(time.perf_counter() - t_start, 1)
    return {
        "cancelled": False,
        "output_path": output_path,
        "frames_processed": frame_idx,
        "blurred_frames": blurred_frames_count,
        "seconds": total_time,
    }

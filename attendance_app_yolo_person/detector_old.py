import os, time, json
import cv2
import numpy as np
from ultralytics import YOLO
from tracker import CentroidTracker

PROOF_DIR = "proof_output"

def load_model(model_name="yolov8n.pt"):
    print("Loading YOLO model:", model_name)
    model = YOLO(model_name)
    return model

def process_first_n_seconds(video_path, model, seconds=5, subsample_every_n_frames=1, save_proof_dir=PROOF_DIR, start_time_sec=0.0, conf_thresh=0.35, imgsz=640):
    os.makedirs(save_proof_dir, exist_ok=True)
    print("Opening video:", video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = int(start_time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames_to_process = int(seconds * fps)

    ct = CentroidTracker(max_disappeared=10)
    counts_per_frame = []
    seen_ids = set()
    max_frame = None
    max_ids_count = 0

    processed = 0
    frame_idx = start_frame
    print(f"Processing {frames_to_process} frames @ {fps} fps (start frame {start_frame})")

    while processed < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        if processed % subsample_every_n_frames == 0:
            results = model(frame, device='cpu', conf=conf_thresh, imgsz=imgsz, verbose=False)
            rects = []
            confs = []

            for res in results:
                if hasattr(res, "boxes"):
                    for box in res.boxes:
                        try:
                            cls = int(box.cls.cpu().numpy())
                        except Exception:
                            try:
                                cls = int(box.cls.numpy())
                            except Exception:
                                cls = int(box.cls)
                        if cls != 0:
                            continue
                        try:
                            xyxy = box.xyxy.cpu().numpy().flatten().tolist()
                        except Exception:
                            try:
                                xyxy = box.xyxy.numpy().flatten().tolist()
                            except Exception:
                                xyxy = list(box.xyxy)

                        if len(xyxy) >= 4:
                            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                            rects.append([x1, y1, x2, y2])
                            try:
                                conf = float(box.conf.cpu().numpy())
                            except Exception:
                                try:
                                    conf = float(box.conf.numpy())
                                except Exception:
                                    conf = float(box.conf)
                            confs.append(conf)

            objects = ct.update(rects)
            current_ids = set(objects.keys())
            seen_ids |= current_ids
            counts_per_frame.append(len(current_ids))

            if len(current_ids) > max_ids_count:
                max_ids_count = len(current_ids)
                max_frame = frame.copy()
                bboxes_ids = []
                rect_centroids = []
                for r in rects:
                    cx = int((r[0] + r[2]) / 2.0)
                    cy = int((r[1] + r[3]) / 2.0)
                    rect_centroids.append((cx, cy))
                for oid, centroid in objects.items():
                    best_i = None
                    best_d = float('inf')
                    for i, rc in enumerate(rect_centroids):
                        d = (rc[0] - centroid[0])**2 + (rc[1] - centroid[1])**2
                        if d < best_d:
                            best_d = d
                            best_i = i
                    if best_i is not None and best_i < len(rects):
                        bboxes_ids.append((rects[best_i], oid, confs[best_i] if best_i < len(confs) else 0.0))

                for (bbox, oid, conf) in bboxes_ids:
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    cv2.rectangle(max_frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(max_frame, f"ID:{oid} {conf:.2f}", (x1, max(0,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        processed += 1
        frame_idx += 1

    cap.release()

    unique_count = len(seen_ids)
    median_count = int(np.median(counts_per_frame)) if counts_per_frame else 0
    robust_count = unique_count if unique_count > 0 else median_count

    proof_path = None
    proof_meta = None
    if max_frame is not None:
        timestamp = int(time.time())
        proof_filename = f"proof_persons_idx{frame_idx}_count{max_ids_count}_{timestamp}.png"
        proof_path = os.path.join(save_proof_dir, proof_filename)
        cv2.putText(max_frame, f"persons={max_ids_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
        cv2.imwrite(proof_path, max_frame)

        proof_meta = {
            "proof_image": proof_filename,
            "frame_index": frame_idx,
            "detected_persons_in_proof_frame": max_ids_count,
            "counts_per_frame": counts_per_frame,
            "unique_tracked_ids_count": unique_count,
            "robust_count": robust_count,
            "seconds_processed": seconds,
            "fps_used": fps if 'fps' in locals() else None,
            "start_time_sec": start_time_sec
        }
        with open(os.path.join(save_proof_dir, proof_filename.replace('.png', '.json')), 'w') as jf:
            json.dump(proof_meta, jf, indent=2)

    return {
        "robust_count": robust_count,
        "counts_per_frame": counts_per_frame,
        "unique_tracked_ids_count": unique_count,
        "proof_image_path": proof_path,
        "proof_metadata": proof_meta
    }

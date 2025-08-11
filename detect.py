"""
#!/usr/bin/env python3
import argparse, csv, json, time
from collections import Counter
from datetime import datetime
from pathlib import Path
import cv2 as cv
import numpy as np

# Optional HEIC support (only used if available)
try:
    from PIL import Image
    from pillow_heif import register_heif
    register_heif()
    _HEIC_OK = True
except Exception:
    _HEIC_OK = False
    Image = None

# ---------- Camera helper (macOS) ----------
def open_cam(force_idx=None):
    if force_idx is not None:
        cap = cv.VideoCapture(force_idx, apiPreference=cv.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {force_idx}")
        print(f"[INFO] Using camera index {force_idx}")
        return cap
    for idx in (0, 1, 2, 3):
        cap = cv.VideoCapture(idx, apiPreference=cv.CAP_AVFOUNDATION)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[INFO] Using camera index {idx}")
                return cap
            cap.release()
    raise RuntimeError("No camera found. Check macOS Settings > Privacy & Security > Camera permissions for Terminal/IDE.")

# ---------- Model builder (two options: tiny or v3@416) ----------
def build_model(root: Path, model: str, size: int, names_path: str = None):
    names_p = Path(names_path) if names_path else (root / "coco.names")

    if model == "tiny":
        cfg_p = root / "yolov3-tiny.cfg"
        weights_p = root / "yolov3-tiny.weights"
        model_name = "yolov3-tiny"
    elif model == "v3":
        # Your yolov3.cfg/weights correspond to the 416 variant
        cfg_p = root / "yolov3.cfg"
        weights_p = root / "yolov3.weights"
        model_name = "yolov3-416"
    else:
        raise ValueError("Model must be 'tiny' or 'v3'.")

    for f in (cfg_p, weights_p, names_p):
        if not Path(f).exists():
            raise FileNotFoundError(f"Missing file: {f}")

    with open(names_p, "r") as f:
        classes = [x.strip() for x in f if x.strip()]

    print(f"[INFO] Loading {model_name}  cfg={cfg_p.name}  weights={weights_p.name}  size={size}")
    net = cv.dnn_DetectionModel(str(cfg_p), str(weights_p))
    net.setInputParams(scale=1/255.0, size=(size, size), swapRB=True)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return net, classes, model_name

# ---------- Draw utility ----------
def draw_dets(img, classes, ids, confs, boxes):
    if len(ids) == 0:
        return img
    for (cid, score, box) in zip(ids.flatten(), confs.flatten(), boxes):
        x, y, w, h = map(int, box)
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{classes[int(cid)]}: {float(score):.2f}"
        cv.putText(img, label, (x, max(15, y-6)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return img

# ---------- Robust image read (supports HEIC if pillow-heif is installed) ----------
def imread_any(path: Path):
    img = cv.imread(str(path))
    if img is not None:
        return img
    if _HEIC_OK and path.suffix.lower() in {".heic", ".heif"}:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
    return None

def main():
    ap = argparse.ArgumentParser(description="YOLOv3-416 / YOLOv3-tiny live & batch detector")
    ap.add_argument("--mode", choices=["live", "images"], default="live", help="Run webcam live or process images")
    ap.add_argument("--model", choices=["tiny", "v3"], default="tiny", help="Select model: tiny (yolov3-tiny) or v3 (yolov3-416)")
    ap.add_argument("--names", help="Path to class names file (default: coco.names next to script)")
    ap.add_argument("--size", type=int, default=416, help="Network input size (e.g., 320, 416, 608)")
    ap.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    ap.add_argument("--nms",  type=float, default=0.30, help="NMS threshold")
    ap.add_argument("--cam",  type=int, default=None, help="Force camera index (0/1/2/3)")

    ap.add_argument("--input","-i", default="photos", help="Input folder or image (images mode)")
    ap.add_argument("--output","-o", default="photos_out", help="Output folder for annotated images (images mode)")
    ap.add_argument("--csv", default="detections.csv", help="CSV filename (written inside output)")
    ap.add_argument("--json", default="detections.json", help="JSON filename (written inside output)")

    # Live snapshot options
    ap.add_argument("--live_out_folder", default="live_photos", help="Folder to save live snapshots")
    ap.add_argument("--live_json", default="output_live_photos.json", help="JSON file to store latest live detection")
    ap.add_argument("--live_interval", type=float, default=5.0, help="Seconds between live snapshots")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    net, classes, model_name = build_model(root, args.model, args.size, names_path=args.names)

    if args.mode == "live":
        cap = open_cam(args.cam)
        win = f"Live — {model_name} (ESC/q to quit, s to save)"
        cv.namedWindow(win, cv.WINDOW_NORMAL)

        live_dir = root / args.live_out_folder
        live_dir.mkdir(parents=True, exist_ok=True)
        live_json_path = root / args.live_json
        last_save = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ids, confs, boxes = net.detect(frame, confThreshold=float(args.conf), nmsThreshold=float(args.nms))
            draw_dets(frame, classes, ids, confs, boxes)
            cv.imshow(win, frame)

            now = time.time()
            should_save = (now - last_save) >= float(args.live_interval)

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('s'):
                should_save = True

            if should_save:
                h, w = frame.shape[:2]
                dets_json = []
                per_counts = Counter()
                if len(ids) > 0:
                    for (cid, score, box) in zip(ids.flatten(), confs.flatten(), boxes):
                        x, y, bw, bh = map(int, box)
                        cls = classes[int(cid)]
                        dets_json.append({
                            "class": cls,
                            "confidence": float(score),
                            "box": {"x": x, "y": y, "w": bw, "h": bh}
                        })
                        per_counts[cls] += 1

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_name = f"live_{ts}.jpg"
                snap_path = live_dir / snap_name
                cv.imwrite(str(snap_path), frame)
                last_save = now

                latest_doc = {
                    "meta": {
                        "model": model_name,
                        "input_size": int(args.size),
                        "conf": float(args.conf),
                        "nms": float(args.nms),
                        "interval_sec": float(args.live_interval),
                        "camera_index": args.cam if args.cam is not None else "auto"
                    },
                    "latest": {
                        "timestamp": datetime.now().astimezone().isoformat(),
                        "file": str(snap_path),
                        "size": {"width": int(w), "height": int(h)},
                        "detections": dets_json,
                        "counts": dict(per_counts)
                    }
                }
                with open(live_json_path, "w") as jf:
                    json.dump(latest_doc, jf, indent=2)
                print(f"[LIVE] Saved {snap_name} and updated {live_json_path.name}")

        cap.release()
        cv.destroyAllWindows()

    else:
        in_path = (root / args.input)
        out_dir = (root / args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect images (allow single file or folder)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
        if in_path.is_file():
            images = [in_path]
        else:
            images = sorted([p for p in in_path.iterdir() if p.suffix.lower() in exts])

        if not images:
            raise SystemExit(f"No images found at {in_path}. Supported: {sorted(exts)}")

        csv_path = out_dir / args.csv
        json_path = out_dir / args.json

        json_doc = {
            "meta": {
                "model": model_name,
                "input_size": int(args.size),
                "conf": float(args.conf),
                "nms": float(args.nms),
                "images_dir": str(in_path if in_path.is_dir() else in_path.parent),
                "heic_supported": bool(_HEIC_OK)
            },
            "images": [],
            "totals": {},
            "num_images": 0
        }
        global_counts = Counter()

        with open(csv_path, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["file", "class", "confidence", "x", "y", "w", "h"])

            for idx, img_p in enumerate(images, 1):
                img = imread_any(img_p)
                if img is None:
                    print(f"[WARN] Could not read {img_p.name}; skipping.")
                    continue

                h, w = img.shape[:2]
                ids, confs, boxes = net.detect(img, confThreshold=float(args.conf), nmsThreshold=float(args.nms))

                dets_json = []
                per_counts = Counter()

                if len(ids) > 0:
                    for (cid, score, box) in zip(ids.flatten(), confs.flatten(), boxes):
                        x, y, bw, bh = map(int, box)
                        cls = classes[int(cid)]
                        dets_json.append({
                            "class": cls,
                            "confidence": float(score),
                            "box": {"x": x, "y": y, "w": bw, "h": bh}
                        })
                        per_counts[cls] += 1
                        writer.writerow([img_p.name, cls, float(score), x, y, bw, bh])

                        cv.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                        label = f"{cls}: {float(score):.2f}"
                        cv.putText(img, label, (x, max(15, y-6)),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                out_p = out_dir / img_p.name
                cv.imwrite(str(out_p), img)

                json_doc["images"].append({
                    "file": img_p.name,
                    "size": {"width": int(w), "height": int(h)},
                    "detections": dets_json,
                    "counts": dict(per_counts)
                })
                global_counts.update(per_counts)

                print(f"[{idx}/{len(images)}] saved -> {out_p.name}")

        json_doc["totals"] = dict(global_counts)
        json_doc["num_images"] = len(json_doc["images"])
        with open(json_path, "w") as jf:
            json.dump(json_doc, jf, indent=2)

        print(f"\nAll done."
              f"\nAnnotated images: {out_dir}"
              f"\nCSV: {csv_path}"
              f"\nJSON: {json_path}")

if __name__ == "__main__":
    main()
"""

#!/usr/bin/env python3
import argparse, csv, json, time
from collections import Counter
from datetime import datetime
from pathlib import Path
import cv2 as cv
import numpy as np

# Optional HEIC support (only used if available)
try:
    from PIL import Image
    from pillow_heif import register_heif
    register_heif()
    _HEIC_OK = True
except Exception:
    _HEIC_OK = False
    Image = None

# ---------- Camera helper (macOS) ----------
def open_cam(force_idx=None):
    if force_idx is not None:
        cap = cv.VideoCapture(force_idx, apiPreference=cv.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {force_idx}")
        print(f"[INFO] Using camera index {force_idx}")
        return cap
    for idx in (0, 1, 2, 3):
        cap = cv.VideoCapture(idx, apiPreference=cv.CAP_AVFOUNDATION)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                print(f"[INFO] Using camera index {idx}")
                return cap
            cap.release()
    raise RuntimeError("No camera found. Check macOS Settings > Privacy & Security > Camera permissions for Terminal/IDE.")

# ---------- Model builder (two options: tiny or v3@416) ----------
def build_model(root: Path, model: str, size: int, names_path: str = None):
    names_p = Path(names_path) if names_path else (root / "coco.names")

    if model == "tiny":
        cfg_p = root / "yolov3-tiny.cfg"
        weights_p = root / "yolov3-tiny.weights"
        model_name = "yolov3-tiny"
    elif model == "v3":
        cfg_p = root / "yolov3.cfg"
        weights_p = root / "yolov3.weights"
        model_name = "yolov3-416"
    else:
        raise ValueError("Model must be 'tiny' or 'v3'.")

    for f in (cfg_p, weights_p, names_p):
        if not Path(f).exists():
            raise FileNotFoundError(f"Missing file: {f}")

    with open(names_p, "r") as f:
        classes = [x.strip() for x in f if x.strip()]

    print(f"[INFO] Loading {model_name}  cfg={cfg_p.name}  weights={weights_p.name}  size={size}")
    net = cv.dnn_DetectionModel(str(cfg_p), str(weights_p))
    net.setInputParams(scale=1/255.0, size=(size, size), swapRB=True)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    return net, classes, model_name

# ---------- Draw utility ----------
def draw_dets(img, classes, ids, confs, boxes):
    if len(ids) == 0:
        return img
    for (cid, score, box) in zip(ids.flatten(), confs.flatten(), boxes):
        x, y, w, h = map(int, box)
        cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"{classes[int(cid)]}: {float(score):.2f}"
        cv.putText(img, label, (x, max(15, y-6)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    return img

# ---------- Robust image read (supports HEIC if pillow-heif is installed) ----------
def imread_any(path: Path):
    img = cv.imread(str(path))
    if img is not None:
        return img
    if _HEIC_OK and path.suffix.lower() in {".heic", ".heif"}:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
    return None

def main():
    ap = argparse.ArgumentParser(description="YOLOv3-416 / YOLOv3-tiny live & batch detector")
    ap.add_argument("--mode", choices=["live", "images"], default="live", help="Run webcam live or process images")
    ap.add_argument("--model", choices=["tiny", "v3"], default="tiny", help="Select model: tiny (yolov3-tiny) or v3 (yolov3-416)")
    ap.add_argument("--names", help="Path to class names file (default: coco.names next to script)")
    ap.add_argument("--size", type=int, default=416, help="Network input size (e.g., 320, 416, 608)")
    ap.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    ap.add_argument("--nms",  type=float, default=0.30, help="NMS threshold")
    ap.add_argument("--cam",  type=int, default=None, help="Force camera index (0/1/2/3)")

    ap.add_argument("--input","-i", default="photos", help="Input folder or image (images mode)")
    ap.add_argument("--output","-o", default="photos_out", help="Output folder for annotated images (images mode)")
    ap.add_argument("--csv", default="detections.csv", help="CSV filename (written inside output)")
    ap.add_argument("--json", default="detections.json", help="JSON filename (written inside output)")

    # Live snapshot options
    ap.add_argument("--live_out_folder", default="live_photos", help="Folder to save live snapshots")
    ap.add_argument("--live_complex_json", default="live_tracking_complex.json",
                    help="Full JSON with meta + latest snapshot details")
    ap.add_argument("--live_simple_json", default="live_tracking_simple.json",
                    help="Simple JSON containing only counts")
    ap.add_argument("--live_interval", type=float, default=5.0, help="Seconds between live snapshots")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    net, classes, model_name = build_model(root, args.model, args.size, names_path=args.names)

    if args.mode == "live":
        cap = open_cam(args.cam)
        win = f"Live — {model_name} (ESC/q to quit, s to save)"
        cv.namedWindow(win, cv.WINDOW_NORMAL)

        live_dir = root / args.live_out_folder
        live_dir.mkdir(parents=True, exist_ok=True)
        complex_json_path = root / args.live_complex_json
        simple_json_path = root / args.live_simple_json
        last_save = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ids, confs, boxes = net.detect(frame, confThreshold=float(args.conf), nmsThreshold=float(args.nms))
            draw_dets(frame, classes, ids, confs, boxes)
            cv.imshow(win, frame)

            now = time.time()
            should_save = (now - last_save) >= float(args.live_interval)

            key = cv.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
            if key == ord('s'):
                should_save = True

            if should_save:
                h, w = frame.shape[:2]
                dets_json = []
                per_counts = Counter()
                if len(ids) > 0:
                    for (cid, score, box) in zip(ids.flatten(), confs.flatten(), boxes):
                        x, y, bw, bh = map(int, box)
                        cls = classes[int(cid)]
                        dets_json.append({
                            "class": cls,
                            "confidence": float(score),
                            "box": {"x": x, "y": y, "w": bw, "h": bh}
                        })
                        per_counts[cls] += 1

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap_name = f"live_{ts}.jpg"
                snap_path = live_dir / snap_name
                cv.imwrite(str(snap_path), frame)
                last_save = now

                latest_doc = {
                    "meta": {
                        "model": model_name,
                        "input_size": int(args.size),
                        "conf": float(args.conf),
                        "nms": float(args.nms),
                        "interval_sec": float(args.live_interval),
                        "camera_index": args.cam if args.cam is not None else "auto"
                    },
                    "latest": {
                        "timestamp": datetime.now().astimezone().isoformat(),
                        "file": str(snap_path),
                        "size": {"width": int(w), "height": int(h)},
                        "detections": dets_json,
                        "counts": dict(per_counts)
                    }
                }
                # Write complex JSON
                with open(complex_json_path, "w") as jf:
                    json.dump(latest_doc, jf, indent=2)
                # Write simple JSON (counts only)
                with open(simple_json_path, "w") as jf:
                    json.dump({"counts": dict(per_counts)}, jf, indent=2)

                print(f"[LIVE] Saved {snap_name} | JSON -> {complex_json_path.name}, {simple_json_path.name}")

        cap.release()
        cv.destroyAllWindows()

    else:
        in_path = (root / args.input)
        out_dir = (root / args.output)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect images (allow single file or folder)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".heic", ".heif"}
        if in_path.is_file():
            images = [in_path]
        else:
            images = sorted([p for p in in_path.iterdir() if p.suffix.lower() in exts])

        if not images:
            raise SystemExit(f"No images found at {in_path}. Supported: {sorted(exts)}")

        csv_path = out_dir / args.csv
        json_path = out_dir / args.json

        # JSON document scaffold
        json_doc = {
            "meta": {
                "model": model_name,
                "input_size": int(args.size),
                "conf": float(args.conf),
                "nms": float(args.nms),
                "images_dir": str(in_path if in_path.is_dir() else in_path.parent),
                "heic_supported": bool(_HEIC_OK)
            },
            "images": [],
            "totals": {},
            "num_images": 0
        }
        global_counts = Counter()

        with open(csv_path, "w", newline="") as fcsv:
            writer = csv.writer(fcsv)
            writer.writerow(["file", "class", "confidence", "x", "y", "w", "h"])

            for idx, img_p in enumerate(images, 1):
                img = imread_any(img_p)
                if img is None:
                    print(f"[WARN] Could not read {img_p.name}; skipping.")
                    continue

                h, w = img.shape[:2]
                ids, confs, boxes = net.detect(img, confThreshold=float(args.conf), nmsThreshold=float(args.nms))

                dets_json = []
                per_counts = Counter()

                if len(ids) > 0:
                    for (cid, score, box) in zip(ids.flatten(), confs.flatten(), boxes):
                        x, y, bw, bh = map(int, box)
                        cls = classes[int(cid)]
                        dets_json.append({
                            "class": cls,
                            "confidence": float(score),
                            "box": {"x": x, "y": y, "w": bw, "h": bh}
                        })
                        per_counts[cls] += 1
                        writer.writerow([img_p.name, cls, float(score), x, y, bw, bh])

                        # draw on image
                        cv.rectangle(img, (x, y), (x+bw, y+bh), (0, 255, 0), 2)
                        label = f"{cls}: {float(score):.2f}"
                        cv.putText(img, label, (x, max(15, y-6)),
                                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

                out_p = out_dir / img_p.name
                cv.imwrite(str(out_p), img)

                json_doc["images"].append({
                    "file": img_p.name,
                    "size": {"width": int(w), "height": int(h)},
                    "detections": dets_json,
                    "counts": dict(per_counts)
                })
                global_counts.update(per_counts)

                print(f"[{idx}/{len(images)}] saved -> {out_p.name}")

        json_doc["totals"] = dict(global_counts)
        json_doc["num_images"] = len(json_doc["images"])
        with open(json_path, "w") as jf:
            json.dump(json_doc, jf, indent=2)

        print(f"\nAll done."
              f"\nAnnotated images: {out_dir}"
              f"\nCSV: {csv_path}"
              f"\nJSON: {json_path}")

if __name__ == "__main__":
    main()

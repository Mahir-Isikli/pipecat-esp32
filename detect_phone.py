#!/usr/bin/env python3
"""
detect_phone.py — Use iPhone camera via Photo Booth / QuickTime window capture, or direct webcam,
and run YOLOv3-tiny / YOLOv3-416 in OpenCV DNN.

Deps for window capture:
  pip install mss pyobjc
macOS: grant Screen Recording permission to your Terminal/IDE when prompted.
"""

import argparse, csv, json, time
from collections import Counter
from datetime import datetime
from pathlib import Path
import cv2 as cv
import numpy as np

# ---------- Optional HEIC support for images mode ----------
try:
    from PIL import Image
    from pillow_heif import register_heif
    register_heif()
    _HEIC_OK = True
except Exception:
    _HEIC_OK = False
    Image = None

# ---------- macOS window APIs + screen capture for QuickTime/Photo Booth ----------
_QT_OK = True
try:
    from Quartz import (
        CGWindowListCopyWindowInfo,
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
    )
    import mss
except Exception:
    _QT_OK = False

# ---------- Webcam helper (macOS) ----------
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

# ---------- Find app window bounds (Photo Booth / QuickTime) ----------
def find_app_window_bounds(app_name, title_hint=None):
    """
    Returns (x, y, w, h) of the largest on-screen window owned by `app_name`.
    Prefers windows whose title contains `title_hint`.
    Coordinates are in screen points; use --scale on Retina (often 2.0).
    """
    if not _QT_OK:
        raise RuntimeError("Window capture needs `pyobjc` and `mss`: pip install pyobjc mss")

    info = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    candidates = []
    for w in info:
        owner = w.get("kCGWindowOwnerName", "")
        name  = w.get("kCGWindowName", "")
        bounds = w.get("kCGWindowBounds", {})
        layer = w.get("kCGWindowLayer", 0)
        if owner == app_name and layer == 0 and bounds:
            x = int(bounds.get("X", 0))
            y = int(bounds.get("Y", 0))
            ww = int(bounds.get("Width", 0))
            hh = int(bounds.get("Height", 0))
            if ww > 50 and hh > 50:
                score = ww * hh
                if title_hint and title_hint.lower() in str(name).lower():
                    score *= 1.5
                candidates.append((score, (x, y, ww, hh), name))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda t: t[0])
    best = candidates[0][1]
    print(f"[INFO] {app_name} window bounds: {best} (title='{candidates[0][2]}')")
    return best  # (x, y, w, h)

# ---------- Model builder (two options: tiny or v3@416) ----------
def build_model(root: Path, model: str, size: int, names_path: str = None):
    names_p = Path(names_path) if names_path else (root / "coco.names")

    if model == "tiny":
        cfg_p = root / "yolov3-tiny.cfg"
        weights_p = root / "yolov3-tiny.weights"
        model_name = "yolov3-tiny"
    elif model == "v3":
        cfg_p = root / "yolov3.cfg"       # your v3@416
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

# ---------- Drawing ----------
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

# ---------- Robust image read (images mode) ----------
def imread_any(path: Path):
    img = cv.imread(str(path))
    if img is not None:
        return img
    if _HEIC_OK and path.suffix.lower() in {".heic", ".heif"}:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
    return None

# ---------- Optional rotation ----------
def rotate_frame(frame, deg):
    if deg == 90:
        return cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv.rotate(frame, cv.ROTATE_180)
    if deg == 270:
        return cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
    return frame

def main():
    ap = argparse.ArgumentParser(description="YOLOv3-416 / YOLOv3-tiny via Photo Booth/QuickTime window, webcam, or images")
    # source selection
    ap.add_argument("--source", choices=["quicktime", "cam", "images"], default="quicktime",
                    help="quicktime: capture Photo Booth/QuickTime window; cam: local webcam; images: process folder/file")

    # model and common detection params
    ap.add_argument("--model", choices=["tiny", "v3"], default="tiny", help="Model: tiny (yolov3-tiny) or v3 (yolov3-416)")
    ap.add_argument("--names", help="Path to class names file (default: coco.names next to script)")
    ap.add_argument("--size", type=int, default=416, help="Network input size (e.g., 320, 416, 608)")
    ap.add_argument("--conf", type=float, default=0.20, help="Confidence threshold")
    ap.add_argument("--nms",  type=float, default=0.30, help="NMS threshold")

    # webcam params
    ap.add_argument("--cam",  type=int, default=None, help="Camera index for --source cam (0/1/2/3)")

    # window capture params
    ap.add_argument("--app_name", default="Photo Booth",
                    help="App window to capture (e.g., 'Photo Booth' or 'QuickTime Player')")
    ap.add_argument("--qt_title_hint", default=None, help="Prefer window whose title contains this text")
    ap.add_argument("--region", nargs=4, type=int, metavar=("X","Y","W","H"),
                    help="Manual capture region in screen points if auto-detect fails")
    ap.add_argument("--scale", type=float, default=1.0, help="Multiply bounds by this (Retina often 2.0)")
    ap.add_argument("--show_crop", action="store_true", help="Draw a blue border to visualize the crop area")
    ap.add_argument("--rotate", type=int, choices=[0,90,180,270], default=0, help="Rotate frames to fix orientation")

    # images mode params
    ap.add_argument("--input","-i", default="photos", help="Input folder or image (images mode)")
    ap.add_argument("--output","-o", default="photos_out", help="Output folder for annotated images (images mode)")
    ap.add_argument("--csv", default="detections.csv", help="CSV filename (written inside output)")
    ap.add_argument("--json", default="detections.json", help="JSON filename (written inside output)")

    # live snapshot options (quicktime/cam)
    ap.add_argument("--live_out_folder", default="live_photos", help="Folder to save live snapshots")
    ap.add_argument("--live_json", default="output_live_photos.json", help="Rolling JSON with latest detections")
    ap.add_argument("--live_interval", type=float, default=30.0, help="Seconds between live snapshots")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    net, classes, model_name = build_model(root, args.model, args.size, names_path=args.names)

    # -------------------- QUICKTIME/PHOTO BOOTH (WINDOW CAPTURE) --------------------
    if args.source == "quicktime":
        if not _QT_OK:
            raise SystemExit("QuickTime/Photo Booth mode requires: pip install pyobjc mss")

        # Determine capture region
        if args.region:
            x, y, w, h = args.region
            print(f"[INFO] Using manual region: {(x,y,w,h)}  scale={args.scale}")
        else:
            bounds = find_app_window_bounds(args.app_name, args.qt_title_hint)
            if not bounds:
                raise SystemExit(f"No '{args.app_name}' window found. Open {args.app_name} and select your iPhone camera.")
            x, y, w, h = bounds
            if args.scale and args.scale != 1.0:
                x = int(x * args.scale); y = int(y * args.scale)
                w = int(w * args.scale); h = int(h * args.scale)
            print(f"[INFO] Capture region (after scale): {(x,y,w,h)}")

        live_dir = root / args.live_out_folder
        live_dir.mkdir(parents=True, exist_ok=True)
        live_json_path = root / args.live_json
        last_save = 0.0

        with mss.mss() as sct:
            win = f"{args.app_name} — {model_name}  [ESC/q quit, s snapshot]"
            cv.namedWindow(win, cv.WINDOW_NORMAL)
            monitor = {"left": x, "top": y, "width": w, "height": h}

            while True:
                sct_img = sct.grab(monitor)  # BGRA ndarray
                frame = np.array(sct_img)
                frame = cv.cvtColor(frame, cv.COLOR_BGRA2BGR)

                if args.rotate:
                    frame = rotate_frame(frame, args.rotate)

                if args.show_crop:
                    cv.rectangle(frame, (1,1), (frame.shape[1]-2, frame.shape[0]-2), (255,0,0), 2)

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
                    h_img, w_img = frame.shape[:2]
                    dets_json, per_counts = [], Counter()
                    if len(ids) > 0:
                        for (cid, score, box) in zip(ids.flatten(), confs.flatten(), boxes):
                            x0, y0, bw, bh = map(int, box)
                            cls = classes[int(cid)]
                            dets_json.append({
                                "class": cls,
                                "confidence": float(score),
                                "box": {"x": x0, "y": y0, "w": bw, "h": bh}
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
                            "source": "window",
                            "app_name": args.app_name,
                            "region": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                            "scale": float(args.scale),
                            "rotate": int(args.rotate)
                        },
                        "latest": {
                            "timestamp": datetime.now().astimezone().isoformat(),
                            "file": str(snap_path),
                            "size": {"width": int(w_img), "height": int(h_img)},
                            "detections": dets_json,
                            "counts": dict(per_counts)
                        }
                    }
                    with open(live_json_path, "w") as jf:
                        json.dump(latest_doc, jf, indent=2)
                    print(f"[LIVE] Saved {snap_name} and updated {live_json_path.name}")

            cv.destroyAllWindows()
        return

    # -------------------- WEBCAM MODE --------------------
    if args.source == "cam":
        cap = open_cam(args.cam)
        win = f"Live — {model_name} (ESC/q quit, s snapshot)"
        cv.namedWindow(win, cv.WINDOW_NORMAL)

        live_dir = root / args.live_out_folder
        live_dir.mkdir(parents=True, exist_ok=True)
        live_json_path = root / args.live_json
        last_save = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                print("[ERR] Failed to read from local camera.")
                break

            if args.rotate:
                frame = rotate_frame(frame, args.rotate)

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
                h_img, w_img = frame.shape[:2]
                dets_json, per_counts = [], Counter()
                if len(ids) > 0:
                    for (cid, score, box) in zip(ids.flatten(), confs.flatten(), boxes):
                        x0, y0, bw, bh = map(int, box)
                        cls = classes[int(cid)]
                        dets_json.append({
                            "class": cls,
                            "confidence": float(score),
                            "box": {"x": x0, "y": y0, "w": bw, "h": bh}
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
                        "source": f"cam:{args.cam if args.cam is not None else 'auto'}",
                        "rotate": int(args.rotate)
                    },
                    "latest": {
                        "timestamp": datetime.now().astimezone().isoformat(),
                        "file": str(snap_path),
                        "size": {"width": int(w_img), "height": int(h_img)},
                        "detections": dets_json,
                        "counts": dict(per_counts)
                    }
                }
                with open(live_json_path, "w") as jf:
                    json.dump(latest_doc, jf, indent=2)
                print(f"[LIVE] Saved {snap_name} and updated {live_json_path.name}")

        try:
            cap.release()
        except Exception:
            pass
        cv.destroyAllWindows()
        return

    # -------------------- IMAGES MODE --------------------
    in_path = (root / args.input)
    out_dir = (root / args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

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
                    x0, y0, bw, bh = map(int, box)
                    cls = classes[int(cid)]
                    dets_json.append({
                        "class": cls,
                        "confidence": float(score),
                        "box": {"x": x0, "y": y0, "w": bw, "h": bh}
                    })
                    per_counts[cls] += 1
                    writer.writerow([img_p.name, cls, float(score), x0, y0, bw, bh])

                    cv.rectangle(img, (x0, y0), (x0+bw, y0+bh), (0, 255, 0), 2)
                    label = f"{cls}: {float(score):.2f}"
                    cv.putText(img, label, (x0, max(15, y0-6)),
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

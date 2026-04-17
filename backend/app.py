"""
Real-Time Facial Emotion Detection – Flask Backend
Run:  python app.py  (from project root)
"""

import os, sys, time, json, base64, threading, collections
import numpy as np
import cv2
from flask import Flask, Response, jsonify, send_from_directory, request

# ── Resolve paths relative to project root ─────────────────────────────────
ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "model", "emotion_model.h5")
FRONTEND   = os.path.join(ROOT, "frontend")

sys.path.insert(0, ROOT)

# ── Lazy-load TF to keep startup fast ──────────────────────────────────────
_model = None
_model_lock = threading.Lock()

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
IMG_SIZE = 48
SMOOTH_FRAMES = 5          # moving-average window
FACE_CASCADE   = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── smoothing/track buffers keyed by stable track id ────────────────────────
smooth_buf: dict = {}
track_boxes: dict = {}
next_track_id = 0

# ── session emotion tracker ────────────────────────────────────────────────
session_counts: dict = {e: 0 for e in EMOTIONS}
session_lock = threading.Lock()
session_start = time.time()

# ── shared frame & result state ────────────────────────────────────────────
_frame_lock   = threading.Lock()
_latest_frame = None          # raw BGR numpy array from webcam
_result_lock  = threading.Lock()
_latest_result = None         # dict with faces list

ENABLE_FACE_MESH = False


def load_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            import tensorflow as tf
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(
                    f"Model not found at {MODEL_PATH}. "
                    "Run  python train_model.py  first."
                )
            _model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[INFO] Model loaded from {MODEL_PATH}")
    return _model


def preprocess(roi_bgr):
    """BGR face crop -> model-ready float32 tensor with lighting normalization."""
    img = cv2.resize(roi_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ycrcb[:, :, 0])
    img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


def smooth_probs(face_id: int, probs: np.ndarray) -> np.ndarray:
    if face_id not in smooth_buf:
        smooth_buf[face_id] = collections.deque(maxlen=SMOOTH_FRAMES)
    smooth_buf[face_id].append(probs)
    return np.mean(smooth_buf[face_id], axis=0)


def _iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (aw * ah) + (bw * bh) - inter
    return inter / union if union > 0 else 0.0


def _assign_track_id(box):
    global next_track_id
    best_id, best_iou = None, 0.0
    for tid, prev_box in track_boxes.items():
        ov = _iou(box, prev_box)
        if ov > best_iou:
            best_iou = ov
            best_id = tid
    if best_id is not None and best_iou >= 0.30:
        track_boxes[best_id] = box
        return best_id
    tid = next_track_id
    next_track_id += 1
    track_boxes[tid] = box
    return tid


def detect_and_predict(frame_bgr):
    """Returns list of face dicts with bbox + smoothed emotion probs."""
    model = load_model()
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h0, w0 = gray.shape[:2]
    detect_scale = 0.75
    small = cv2.resize(gray, (int(w0 * detect_scale), int(h0 * detect_scale)), interpolation=cv2.INTER_AREA)
    # Equalize histogram for better detection in varied lighting
    gray_eq = cv2.equalizeHist(small)
    faces = FACE_CASCADE.detectMultiScale(
        gray_eq, scaleFactor=1.09, minNeighbors=3, minSize=(48, 48)
    )
    result_faces = []
    current_ids = set()
    for (x, y, w, h) in faces:
        x = int(x / detect_scale)
        y = int(y / detect_scale)
        w = int(w / detect_scale)
        h = int(h / detect_scale)
        pad = int(0.1 * min(w, h))
        x1  = max(0, x - pad)
        y1  = max(0, y - pad)
        x2  = min(frame_bgr.shape[1], x + w + pad)
        y2  = min(frame_bgr.shape[0], y + h + pad)
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        track_id = _assign_track_id((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        current_ids.add(track_id)
        tensor = preprocess(roi)
        # Test-time augmentation for stability: average original + mirrored crop
        tensor_flip = tensor[:, :, ::-1, :]
        p1 = model.predict(tensor, verbose=0)[0]
        p2 = model.predict(tensor_flip, verbose=0)[0]
        raw_probs = (p1 + p2) / 2.0
        probs = smooth_probs(track_id, raw_probs)
        dom_idx   = int(np.argmax(probs))
        dom_emotion = EMOTIONS[dom_idx]

        # update session counts
        with session_lock:
            session_counts[dom_emotion] += 1

        result_faces.append({
            "id":        track_id,
            "bbox":      [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "dominant":  dom_emotion,
            "confidence": float(round(probs[dom_idx] * 100, 1)),
            "probs":     {EMOTIONS[i]: float(round(probs[i] * 100, 1))
                          for i in range(len(EMOTIONS))},
        })
    stale_ids = [tid for tid in list(track_boxes.keys()) if tid not in current_ids]
    for tid in stale_ids:
        track_boxes.pop(tid, None)
        smooth_buf.pop(tid, None)
    return result_faces


# ── Webcam capture + inference threads (decoupled) ─────────────────────────
_cap = None
_capture_thread = None
_inference_thread = None
_running = False

# A small queue for frames to be processed by the inference thread
import queue as _queue
_infer_queue = _queue.Queue(maxsize=2)   # non-blocking; drop stale frames


def _capture_loop():
    """Reads frames as fast as possible; never blocks on inference."""
    global _cap, _latest_frame, _running
    _cap = cv2.VideoCapture(0)
    _cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    _cap.set(cv2.CAP_PROP_FPS, 30)
    # Reduce internal buffer so we always get the freshest frame
    _cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while _running:
        ok, frame = _cap.read()
        if not ok:
            time.sleep(0.02)
            continue
        frame = cv2.flip(frame, 1)   # mirror

        with _frame_lock:
            _latest_frame = frame.copy()

        # Push frame for inference; discard if inference is already busy
        try:
            _infer_queue.put_nowait(frame)
        except _queue.Full:
            pass   # skip frame – inference is lagging, keep reading

    if _cap:
        _cap.release()
        _cap = None


def _inference_loop():
    """Dedicated thread: pulls frames from queue and runs model inference."""
    global _latest_result, _running
    INFER_INTERVAL = 0.07   # minimum seconds between consecutive predictions
    last_run = 0

    while _running:
        try:
            frame = _infer_queue.get(timeout=0.5)
        except _queue.Empty:
            continue

        now = time.time()
        if now - last_run < INFER_INTERVAL:
            continue   # rate-limit without blocking the capture thread
        last_run = now

        try:
            faces = detect_and_predict(frame)
        except Exception as e:
            faces = []
            print(f"[WARN] Inference error: {e}")

        with _result_lock:
            _latest_result = {"timestamp": now, "faces": faces}


def start_capture():
    global _capture_thread, _inference_thread, _running
    if _capture_thread and _capture_thread.is_alive():
        return
    try:
        load_model()   # warm up model before streaming starts
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
    _running = True
    _capture_thread = threading.Thread(target=_capture_loop, daemon=True, name="CaptureThread")
    _inference_thread = threading.Thread(target=_inference_loop, daemon=True, name="InferenceThread")
    _capture_thread.start()
    _inference_thread.start()
    print("[INFO] Webcam capture + inference threads started.")


def stop_capture():
    """Gracefully stop the capture and inference threads."""
    global _running, _cap, _latest_frame, _latest_result
    _running = False
    # Drain the queue so inference thread exits quickly
    while not _infer_queue.empty():
        try:
            _infer_queue.get_nowait()
        except _queue.Empty:
            break
    _latest_frame = None
    with _result_lock:
        _latest_result = None
    smooth_buf.clear()
    track_boxes.clear()
    print("[INFO] Webcam capture stopped.")


# ─── Emotion color map for OpenCV (BGR) ────────────────────────────────────
EMOTION_BGR = {
    "Angry":    (64,  64,  255),   # red
    "Disgust":  (255, 68,  170),   # purple
    "Fear":     (0,   144, 255),   # orange
    "Happy":    (176, 255, 0  ),   # green
    "Neutral":  (255, 207, 0  ),   # blue
    "Sad":      (255, 136, 85 ),   # blue
    "Surprise": (0,   224, 255),   # yellow
}


# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder=FRONTEND)


@app.route("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONTEND, filename)


@app.route("/api/status")
def status():
    model_ready = os.path.exists(MODEL_PATH)
    return jsonify({
        "model_ready":   model_ready,
        "model_path":    MODEL_PATH,
        "capture_alive": _capture_thread is not None and _capture_thread.is_alive(),
        "mediapipe":     ENABLE_FACE_MESH,
    })


@app.route("/api/emotions")
def emotions_api():
    """Return latest emotion predictions as JSON."""
    with _result_lock:
        result = _latest_result
    if result is None:
        return jsonify({"faces": [], "timestamp": time.time()})
    return jsonify(result)


@app.route("/api/session")
def session_api():
    """Return cumulative session emotion counts and dominant emotion."""
    with session_lock:
        counts = dict(session_counts)
    total = sum(counts.values())
    percentages = {e: round(counts[e] / total * 100, 1) if total > 0 else 0
                   for e in EMOTIONS}
    dominant = max(counts, key=counts.get) if total > 0 else "Neutral"
    elapsed = int(time.time() - session_start)
    return jsonify({
        "counts":     counts,
        "percentages": percentages,
        "dominant":   dominant,
        "total":      total,
        "elapsed_s":  elapsed,
    })


@app.route("/api/session/reset", methods=["POST"])
def session_reset():
    global session_start
    with session_lock:
        for e in EMOTIONS:
            session_counts[e] = 0
    session_start = time.time()
    return jsonify({"ok": True})


@app.route("/api/capture/stop", methods=["POST"])
def capture_stop():
    stop_capture()
    return jsonify({"ok": True, "running": False})


@app.route("/api/capture/start", methods=["POST"])
def capture_start():
    start_capture()
    return jsonify({"ok": True, "running": True})


@app.route("/api/capture/status")
def capture_status():
    return jsonify({"running": _running})


@app.route("/video_feed")
def video_feed():
    """MJPEG stream with face bounding boxes."""
    def gen():
        while True:
            with _frame_lock:
                frame = _latest_frame
            if frame is None:
                time.sleep(0.03)
                continue

            vis = frame.copy()

            with _result_lock:
                result = _latest_result

            if result:
                for face in result["faces"]:
                    x, y, w, h = face["bbox"]
                    dom   = face["dominant"]
                    conf  = face["confidence"]
                    color = EMOTION_BGR.get(dom, (0, 255, 150))

                    # bounding box
                    cv2.rectangle(vis, (x, y), (x + w, y + h), color, 1)

                    # corner accents
                    tl, thick = 18, 3
                    cv2.line(vis, (x, y),           (x + tl, y),       color, thick)
                    cv2.line(vis, (x, y),           (x, y + tl),       color, thick)
                    cv2.line(vis, (x + w, y),       (x + w - tl, y),   color, thick)
                    cv2.line(vis, (x + w, y),       (x + w, y + tl),   color, thick)
                    cv2.line(vis, (x, y + h),       (x + tl, y + h),   color, thick)
                    cv2.line(vis, (x, y + h),       (x, y + h - tl),   color, thick)
                    cv2.line(vis, (x + w, y + h),   (x + w - tl, y + h), color, thick)
                    cv2.line(vis, (x + w, y + h),   (x + w, y + h - tl), color, thick)

                    # label background + text
                    label = f"{dom}  {conf:.0f}%"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)
                    cv2.rectangle(vis, (x, y - th - 12), (x + tw + 10, y - 2), (0, 0, 0), -1)
                    cv2.putText(vis, label, (x + 5, y - 5),
                                cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1)

            # HUD overlay
            ts = time.strftime("%H:%M:%S")
            cv2.putText(vis, f"EMOTION-AI  {ts}",
                        (10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.48, (0, 255, 150), 1)
            cv2.putText(vis, "NEURAL SCAN ACTIVE",
                        (10, vis.shape[0] - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.42, (0, 200, 255), 1)

            _, jpg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" +
                   jpg.tobytes() + b"\r\n")
            time.sleep(0.025)   # ~40 fps target

            # If capture was stopped, emit a blank pause frame to keep stream alive
            if not _running:
                time.sleep(0.1)

    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    print("=" * 60)
    print("  🧠  Real-Time Facial Emotion Detection")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 60)
    start_capture()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

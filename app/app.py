import base64
import importlib
import sys
import tempfile
from pathlib import Path

import cv2
import gdown
import joblib
import mediapipe as mp
import numpy as np
import streamlit as st
import torch

# ------------------------------------------------------------------
# 1. PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(page_title="Eye Tracking & Emotion AI", layout="wide")
st.title("👁️ Dynamic Eye Tracking & Emotion Recognition")
st.caption(
    "Real-time eye-part segmentation overlay (UNet / MediaPipe) "
    "& sequence-based emotion classification (LSTM)"
)

# ------------------------------------------------------------------
# 2. PATHS, CONSTANTS, DOWNLOADS
# ------------------------------------------------------------------
CURRENT_DIR = Path(__file__).parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

WEIGHTS_SEG = CURRENT_DIR / "best_unet_model.pth"
WEIGHTS_GAZE = CURRENT_DIR / "best_gaze_model.pth"
WEIGHTS_EMOTION = CURRENT_DIR / "best_emotion_lstm.pth"
SCALER_FILE = CURRENT_DIR / "gaze_scaler.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTION_CLASSES = ["Neutral", "Frustrated", "Bored", "Confident"]

# Label ids used internally: 0 = background, 1 = sclera, 2 = iris, 3 = pupil
# Colors are BGR (OpenCV order).
LABEL_NAMES = {1: "Sclera", 2: "Iris", 3: "Pupil"}
LABEL_COLORS_BGR = {
    1: (0, 255, 0),      # green
    2: (255, 255, 0),    # cyan
    3: (255, 0, 255),    # magenta
}
LABEL_COLORS_HEX = {1: "#00ff00", 2: "#00ffff", 3: "#ff00ff"}

# MediaPipe FaceMesh landmark groups
LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
IRIS_A = [468, 469, 470, 471, 472]  # centre first
IRIS_B = [473, 474, 475, 476, 477]
COMBINED_EYE_IDX = [
    33, 133, 160, 159, 158, 144, 145, 153,
    362, 263, 387, 386, 385, 373, 374, 380,
]

FILE_IDS = {
    WEIGHTS_SEG: "11dayKwl4X3UUfERRpyl6s-nz_YXAZvcA",
    WEIGHTS_GAZE: "1EvaC29K0VoCsc7xG72j571cz6mumlsU7",
    WEIGHTS_EMOTION: "1Wh4Rro4jkj9_xCoTs1G11ZA5pNVUPMr7",
    SCALER_FILE: "1uOtZmD7900j5hbSfV4WTJ8DVvec7B-0r",
}


@st.cache_resource(show_spinner="Downloading model files...")
def ensure_files():
    for dest, file_id in FILE_IDS.items():
        if not dest.exists():
            gdown.download(f"https://drive.google.com/uc?id={file_id}", str(dest), quiet=False)
    return True


ensure_files()


# ------------------------------------------------------------------
# 3. LOADERS (cached so Streamlit reruns don't rebuild them)
# ------------------------------------------------------------------
@st.cache_resource
def get_face_mesh():
    module = getattr(mp.solutions, "face_mesh", None)
    if module is None:
        module = importlib.import_module("mediapipe.python.solutions.face_mesh")
    return module.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,  # required for iris landmarks 468-477
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )


@st.cache_resource
def load_all_assets():
    try:
        from Models.emotion_model import EmotionLSTM
        from Models.eyesegementation_model import UNet
        from Models.gaze_model import GazeModel

        seg_model = UNet().to(DEVICE)
        seg_model.load_state_dict(torch.load(WEIGHTS_SEG, map_location=DEVICE))
        seg_model.eval()

        # First conv layer tells us how many input channels the UNet expects
        in_ch = 3
        for m in seg_model.modules():
            if isinstance(m, torch.nn.Conv2d):
                in_ch = m.in_channels
                break

        gaze_model = GazeModel().to(DEVICE)
        gaze_model.load_state_dict(torch.load(WEIGHTS_GAZE, map_location=DEVICE))
        gaze_model.eval()

        emotion_model = EmotionLSTM(input_size=3, num_classes=4).to(DEVICE)
        emotion_model.load_state_dict(torch.load(WEIGHTS_EMOTION, map_location=DEVICE), strict=False)
        emotion_model.eval()

        scaler = joblib.load(SCALER_FILE)
        return seg_model, gaze_model, emotion_model, scaler, in_ch, None
    except Exception as e:  # noqa: BLE001
        return None, None, None, None, 3, str(e)


face_mesh = get_face_mesh()
seg_model, gaze_model, emotion_model, gaze_scaler, expected_in_channels, load_error = load_all_assets()

# ------------------------------------------------------------------
# 4. SIDEBAR SETTINGS
# ------------------------------------------------------------------
st.sidebar.header("⚙️ Settings")
seg_source = st.sidebar.selectbox(
    "Eye-part source",
    ["Auto (UNet, fallback to MediaPipe)", "UNet only", "MediaPipe landmarks only"],
    help="MediaPipe landmarks give a guaranteed colored overlay even if the UNet output is poor.",
)
alpha = st.sidebar.slider("Overlay opacity", 0.1, 1.0, 0.6, 0.05)
binary_thr = st.sidebar.slider("UNet threshold (single-channel models)", 0.05, 0.95, 0.5, 0.05)
has_bg_channel = st.sidebar.checkbox(
    "UNet has a background channel (class 0)",
    value=True,
    help="Turn off if your multi-class UNet outputs only [sclera, iris, pupil].",
)
max_width = st.sidebar.slider("Max processing width (px)", 480, 1920, 960, 80)

# ------------------------------------------------------------------
# 5. FACE / EYE GEOMETRY
# ------------------------------------------------------------------
def detect_landmarks(frame):
    """Return (N, 2) pixel landmarks or None."""
    h, w = frame.shape[:2]
    try:
        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    except Exception:  # noqa: BLE001
        return None
    if not res.multi_face_landmarks:
        return None
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)


def combined_eye_box(pts, h, w):
    """Box around both eyes (used for gaze model + display)."""
    if pts is not None:
        p = pts[COMBINED_EYE_IDX]
        pad_x, pad_y = int(w * 0.02), int(h * 0.015)
        x0 = max(0, int(p[:, 0].min()) - pad_x)
        x1 = min(w, int(p[:, 0].max()) + pad_x)
        y0 = max(0, int(p[:, 1].min()) - pad_y)
        y1 = min(h, int(p[:, 1].max()) + pad_y)
        if x1 - x0 > 20 and y1 - y0 > 20:
            return x0, y0, x1 - x0, y1 - y0
    return int(w * 0.15), int(h * 0.32), int(w * 0.70), int(h * 0.18)


def single_eye_box(pts, contour_idx, h, w, scale=1.6):
    """Roughly square box centred on ONE eye."""
    p = pts[contour_idx]
    x0, y0 = p.min(axis=0)
    x1, y1 = p.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0, 12) * scale
    bx0, bx1 = int(max(0, cx - side / 2)), int(min(w, cx + side / 2))
    by0, by1 = int(max(0, cy - side / 2)), int(min(h, cy + side / 2))
    return bx0, by0, bx1 - bx0, by1 - by0


# ------------------------------------------------------------------
# 6. SEGMENTATION BACKENDS -> label map (0 bg, 1 sclera, 2 iris, 3 pupil)
# ------------------------------------------------------------------
def unet_labels_for_crop(crop):
    ch, cw = crop.shape[:2]
    resized = cv2.resize(crop, (256, 256))
    if expected_in_channels == 1:
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        arr = gray[None].astype(np.float32) / 255.0
    else:
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        arr = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = seg_model(tensor)
        if out.shape[1] > 1:
            pred = torch.argmax(out, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            if not has_bg_channel:
                pred = pred + 1  # shift [sclera, iris, pupil] -> [1, 2, 3]
        else:
            prob = torch.sigmoid(out).squeeze().cpu().numpy()
            pred = (prob > binary_thr).astype(np.uint8)  # single class -> "sclera" color
    pred = np.where(pred > 3, 0, pred).astype(np.uint8)
    return cv2.resize(pred, (cw, ch), interpolation=cv2.INTER_NEAREST)


def unet_label_map(frame, pts):
    h, w = frame.shape[:2]
    label_map = np.zeros((h, w), dtype=np.uint8)
    if pts is not None:
        boxes = [single_eye_box(pts, LEFT_EYE_CONTOUR, h, w), single_eye_box(pts, RIGHT_EYE_CONTOUR, h, w)]
    else:
        boxes = [combined_eye_box(None, h, w)]
    for (x, y, bw, bh) in boxes:
        if bw < 10 or bh < 10:
            continue
        label_map[y:y + bh, x:x + bw] = unet_labels_for_crop(frame[y:y + bh, x:x + bw])
    return label_map


def mediapipe_label_map(frame, pts):
    h, w = frame.shape[:2]
    label_map = np.zeros((h, w), dtype=np.uint8)
    if pts is None:
        return label_map

    for contour_idx, iris_idx in ((LEFT_EYE_CONTOUR, IRIS_A), (RIGHT_EYE_CONTOUR, IRIS_B)):
        eye_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(eye_mask, [pts[contour_idx].astype(np.int32)], 1)

        label_map[eye_mask == 1] = 1  # sclera = whole eye opening

        if pts.shape[0] > max(iris_idx):
            center = pts[iris_idx[0]]
            radius = float(np.mean(np.linalg.norm(pts[iris_idx[1:]] - center, axis=1)))
            cx, cy = int(center[0]), int(center[1])

            iris_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(iris_mask, (cx, cy), max(1, int(radius)), 1, -1)
            label_map[(iris_mask == 1) & (eye_mask == 1)] = 2

            pupil_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(pupil_mask, (cx, cy), max(1, int(radius * 0.4)), 1, -1)
            label_map[(pupil_mask == 1) & (eye_mask == 1)] = 3
    return label_map


def build_label_map(frame, pts):
    if seg_source.startswith("MediaPipe"):
        return mediapipe_label_map(frame, pts)
    unet_map = unet_label_map(frame, pts)
    if seg_source == "UNet only":
        return unet_map
    # Auto: use UNet, but fall back to landmarks when UNet found (almost) nothing
    if np.count_nonzero(unet_map) < 30:
        return mediapipe_label_map(frame, pts)
    return unet_map


def colorize(frame, label_map, opacity):
    color_layer = np.zeros_like(frame)
    for k, c in LABEL_COLORS_BGR.items():
        color_layer[label_map == k] = c
    blended = cv2.addWeighted(frame, 1 - opacity, color_layer, opacity, 0)
    out = frame.copy()
    m = label_map > 0
    out[m] = blended[m]  # assign explicitly (no in-place cv2 write into a slice)
    return out


def draw_legend(frame):
    x, y = 10, 10
    for k, name in LABEL_NAMES.items():
        cv2.rectangle(frame, (x, y), (x + 18, y + 18), LABEL_COLORS_BGR[k], -1)
        cv2.putText(frame, name, (x + 26, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26


# ------------------------------------------------------------------
# 7. FRAME PROCESSING
# ------------------------------------------------------------------
def process_frame(frame, sequence_buffer, frame_count, last_emotion):
    h, w = frame.shape[:2]
    pts = detect_landmarks(frame)
    ex, ey, ew, eh = combined_eye_box(pts, h, w)

    gaze_x, gaze_y, pupil_ratio = 0.5, 0.5, 0.33
    counts = {1: 0, 2: 0, 3: 0}

    # A. Eye-part segmentation
    label_map = build_label_map(frame, pts)
    for k in counts:
        counts[k] = int(np.count_nonzero(label_map == k))
    frame = colorize(frame, label_map, alpha)

    region_area = max(1, ew * eh)
    pupil_ratio = float(np.clip(counts[3] / region_area, 0.05, 0.8)) if counts[3] else 0.33

    # B. Gaze
    eye_crop = frame[ey:ey + eh, ex:ex + ew]
    if eye_crop.shape[0] > 10 and eye_crop.shape[1] > 10:
        rgb = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
        arr = cv2.resize(rgb, (64, 64)).transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            coords = gaze_model(tensor).squeeze().cpu().tolist()
        if isinstance(coords, list) and len(coords) >= 2:
            gaze_x, gaze_y = float(coords[0]), float(coords[1])

    cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 200, 0), 1)
    if pts is None:
        cv2.putText(frame, "No face detected", (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    draw_legend(frame)

    # C. Emotion (every 5 frames once 30 steps are buffered)
    sequence_buffer.append([gaze_x, gaze_y, pupil_ratio])
    if len(sequence_buffer) > 30:
        sequence_buffer.pop(0)

    emotion = last_emotion
    if len(sequence_buffer) == 30 and frame_count % 5 == 0:
        raw = np.asarray(sequence_buffer, dtype=np.float32)
        scaled = gaze_scaler.transform(raw) if gaze_scaler is not None else raw
        seq = torch.from_numpy(np.asarray(scaled, dtype=np.float32)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            idx = int(torch.argmax(emotion_model(seq), dim=1).item())
        emotion = EMOTION_CLASSES[idx]

    return frame, gaze_x, gaze_y, emotion, counts


# ------------------------------------------------------------------
# 8. UI
# ------------------------------------------------------------------
if load_error:
    st.error(f"Error loading models or scaler: {load_error}")
    st.stop()

col_left, col_right = st.columns([2, 1])

with col_right:
    st.subheader("📊 Live Pipeline Metrics")
    emotion_metric = st.empty()
    gaze_metric = st.empty()
    count_metric = st.empty()

    st.markdown("---")
    st.markdown("### 🎨 Eye-Part Color Legend")
    for k, name in LABEL_NAMES.items():
        st.markdown(
            f"<span style='display:inline-block;width:14px;height:14px;background:{LABEL_COLORS_HEX[k]};"
            f"border-radius:3px;margin-right:8px;vertical-align:middle'></span> **{name}**",
            unsafe_allow_html=True,
        )

with col_left:
    st.subheader("🎥 Dynamic AI Video Feed")
    uploaded_video = st.file_uploader("Upload Input Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
    video_placeholder = st.empty()

if uploaded_video is not None:
    if st.button("▶️ Start Real-Time AI Stream", type="primary"):
        suffix = Path(uploaded_video.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_video.getvalue())
            temp_path = Path(tmp.name)

        cap = cv2.VideoCapture(str(temp_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        progress = col_left.progress(0.0)

        sequence_buffer, frame_count, emotion = [], 0, "Analyzing..."

        try:
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                frame_count += 1

                fh, fw = frame.shape[:2]
                if fw > max_width:
                    scale = max_width / fw
                    frame = cv2.resize(frame, (max_width, int(fh * scale)))

                out, gx, gy, emotion, counts = process_frame(frame, sequence_buffer, frame_count, emotion)

                # Send the frame as JPEG bytes: avoids Streamlit's media-file cache
                # deleting the image before the browser fetches it (broken image icon).
                ok_enc, jpg = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok_enc:
                    b64 = base64.b64encode(jpg.tobytes()).decode("ascii")
                    # Embed the frame directly in HTML (data URI) so the browser never
                    # has to fetch a separate media file -> no broken-image icon.
                    video_placeholder.markdown(
                        f'<img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:6px;">',
                        unsafe_allow_html=True,
                    )
                emotion_metric.metric("🧠 Emotion Prediction (LSTM)", emotion)
                gaze_metric.code(f"Gaze Vector (X, Y):\n({gx:.2f}, {gy:.2f})")
                count_metric.code(
                    "Pixels detected:\n"
                    f"  Sclera: {counts[1]}\n  Iris:   {counts[2]}\n  Pupil:  {counts[3]}"
                )
                progress.progress(min(frame_count / total, 1.0))
        finally:
            cap.release()
            temp_path.unlink(missing_ok=True)

        st.success("✅ AI Stream Complete!")
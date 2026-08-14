import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import time
import torch
import sys
import logging
import joblib
import gdown
from pathlib import Path
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode
)

# Streamlit-WebRTC Logs Suppress
logging.getLogger("streamlit_webrtc").setLevel(logging.CRITICAL)

# --- 📁 PATHS MANAGEMENT & MODEL IMPORTS ---
ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

WEIGHTS_SEG = ROOT_DIR / "best_unet_model.pth"
WEIGHTS_GAZE = ROOT_DIR / "best_gaze_model.pth"
WEIGHTS_EMOTION = ROOT_DIR / "best_emotion_lstm.pth"
SCALER_FILE = ROOT_DIR / "gaze_scaler.pkl"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 📥 GOOGLE DRIVE AUTO-DOWNLOADER ---
def download_file_from_google_drive(file_id, destination):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, str(destination), quiet=False)

SEG_FILE_ID = "11dayKwl4X3UUfERRpyl6s-nz_YXAZvcA"       
GAZE_FILE_ID = "1EvaC29K0VoCsc7xG72j571cz6mumlsU7"          
EMOTION_FILE_ID = "1Wh4Rro4jkj9_xCoTs1G11ZA5pNVUPMr7"  
SCALER_FILE_ID = "1uOtZmD7900j5hbSfV4WTJ8DVvec7B-0r"

with st.spinner("Syncing Cloud Architecture... Checking Weights & Scaler..."):
    if not WEIGHTS_SEG.exists():
        download_file_from_google_drive(SEG_FILE_ID, WEIGHTS_SEG)
    if not WEIGHTS_GAZE.exists():
        download_file_from_google_drive(GAZE_FILE_ID, WEIGHTS_GAZE)
    if not WEIGHTS_EMOTION.exists():
        download_file_from_google_drive(EMOTION_FILE_ID, WEIGHTS_EMOTION)
    if not SCALER_FILE.exists():
        download_file_from_google_drive(SCALER_FILE_ID, SCALER_FILE)

try:
    from Models.eyesegementation_model import UNet
    from Models.gaze_model import GazeModel
    from Models.emotion_model import EmotionLSTM  
    models_imported = True
except ImportError as e:
    models_imported = False
    st.error(f"⚠️ Model classes import error: {e}")

@st.cache_resource
def load_vision_models():
    seg, gaze, emotion, scaler = None, None, None, None
    if models_imported:
        if WEIGHTS_SEG.exists():
            seg = UNet().to(DEVICE); seg.eval()
        if WEIGHTS_GAZE.exists():
            gaze = GazeModel().to(DEVICE); gaze.eval()
        if WEIGHTS_EMOTION.exists():
            try:
                emotion = EmotionLSTM(input_size=3, num_classes=4).to(DEVICE); emotion.eval()
            except Exception: pass
        if SCALER_FILE.exists():
            try:
                scaler = joblib.load(SCALER_FILE)
            except Exception: pass
    return seg, gaze, emotion, scaler

seg_model, gaze_model, emotion_model, gaze_scaler = load_vision_models()
EMOTION_CLASSES = ["Neutral", "Frustrated", "Bored", "Confident"]

# --- 💻 CORE UNET PIPELINE PROCESSING FUNCTION ---
def local_process_frame(frame, gaze_history_buffer):
    frame = cv2.resize(frame, (640, 480))
    h, w, _ = frame.shape

    # Focus Eye Zone
    crop_x1, crop_y1 = int(w * 0.15), int(h * 0.15)
    crop_x2, crop_y2 = int(w * 0.85), int(h * 0.65)
    eye_zone = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    gaze_vectors = [0.5, 0.5]
    pupil_size = 0.33
    bbox_coords = None
    eye_detected = False
    detected_emotion = "Calculating..."

    # 1. UNet Segmentation Mask & Contour Bounding Box
    if seg_model is not None and eye_zone.size > 0:
        try:
            img_t = cv2.resize(eye_zone, (256, 256)).transpose((2, 0, 1)) / 255.0
            img_t = torch.tensor([img_t], dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                seg_out = seg_model(img_t)
                if seg_out.shape[1] > 1:
                    pred_mask = torch.argmax(seg_out, dim=1).squeeze().cpu().numpy()
                else:
                    pred_mask = (torch.sigmoid(seg_out).squeeze().cpu().numpy() > 0.5).astype(np.uint8)

                zh, zw, _ = eye_zone.shape
                mask_resized = cv2.resize(pred_mask.astype(np.uint8), (zw, zh), interpolation=cv2.INTER_NEAREST)

                eye_binary = (mask_resized > 0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(eye_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    c = max(contours, key=cv2.contourArea)
                    if cv2.contourArea(c) > 30:
                        bx, by, bw, bh = cv2.boundingRect(c)
                        bbox_coords = (crop_x1 + bx, crop_y1 + by, bw, bh)
                        eye_detected = True

                pupil_pixels = np.sum(mask_resized == 3) if seg_out.shape[1] > 1 else np.sum(mask_resized == 1)
                pupil_size = float(np.clip(pupil_pixels / (zw * zh), 0.05, 0.8))

                color_mask = np.zeros_like(eye_zone, dtype=np.uint8)
                if seg_out.shape[1] > 1:
                    color_mask[mask_resized == 1] = [0, 255, 0]    # Green (Sclera)
                    color_mask[mask_resized == 2] = [255, 255, 0]  # Cyan (Iris)
                    color_mask[mask_resized == 3] = [255, 0, 255]  # Magenta (Pupil)
                else:
                    color_mask[mask_resized == 1] = [180, 105, 255]

                overlay = eye_zone.copy()
                has_features = np.any(color_mask > 0, axis=-1)
                overlay[has_features] = color_mask[has_features]
                cv2.addWeighted(overlay, 0.65, eye_zone, 0.35, 0, frame[crop_y1:crop_y2, crop_x1:crop_x2])
        except Exception:
            pupil_size = 0.33

    # 2. Draw Green Bounding Box
    if bbox_coords is not None:
        bx, by, bw, bh = bbox_coords
        pad = 4
        cv2.rectangle(
            frame, 
            (max(0, bx - pad), max(0, by - pad)), 
            (min(w, bx + bw + pad), min(h, by + bh + pad)), 
            (0, 255, 0), 
            2
        )

    # 3. Gaze Estimation
    if gaze_model is not None and eye_zone.size > 0:
        try:
            gaze_input = cv2.resize(eye_zone, (64, 64)).transpose((2, 0, 1)) / 255.0
            gaze_input = torch.tensor([gaze_input], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                gaze_out = gaze_model(gaze_input)
                gaze_vectors = gaze_out.squeeze().cpu().tolist()
        except Exception: pass

    # 4. History Buffer & LSTM Emotion Prediction
    current_features = [float(gaze_vectors[0]), float(gaze_vectors[1]), float(pupil_size)]
    gaze_history_buffer.append(current_features)
    if len(gaze_history_buffer) > 30:
        gaze_history_buffer.pop(0)

    if emotion_model is not None and len(gaze_history_buffer) == 30:
        try:
            raw_seq = np.array(gaze_history_buffer, dtype=np.float32)
            scaled_seq = gaze_scaler.transform(raw_seq) if gaze_scaler is not None else raw_seq
            seq_tensor = torch.tensor([scaled_seq], dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                emotion_out = emotion_model(seq_tensor)
                pred_idx = torch.argmax(emotion_out, dim=1).item()
                detected_emotion = EMOTION_CLASSES[pred_idx]
        except Exception:
            detected_emotion = "Neutral"

    # UI Visual Overlays
    cv2.putText(frame, f"Emotion: {detected_emotion}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Gaze: ({gaze_vectors[0]:.2f}, {gaze_vectors[1]:.2f})", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return frame, gaze_vectors, eye_detected, detected_emotion


# --- 🎥 WEBRTC WORKER CLASS ---
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
            {"urls": ["stun:global.stun.twilio.com:3478"]}
        ]
    }
)

class EyeTrackerVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.gaze_history = []

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_frame, _, _, _ = local_process_frame(img, self.gaze_history)
        return frame.from_ndarray(processed_frame, format="bgr24")


# --- 🖥️ STREAMLIT UI LAYOUT ---
st.set_page_config(page_title="Vision AI Production Node", layout="wide")

st.markdown("""
    <style>
    .element-container iframe {
        max-width: 540px !important;
        margin: 0 auto !important;
        display: block !important;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("👁️ Enterprise Eye Segmentation, Gaze & Emotion AI Node")

tab_live, tab_video = st.tabs(["📡 Continuous Live Feed", "🎥 File Video Analyzer"])

with tab_live:
    st.subheader("🔴 Real-Time WebRTC Eye Tracking & Segmentation")
    st.write("Click **START** to open camera feed.")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        webrtc_streamer(
            key="eye-tracker-live-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=EyeTrackerVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

with tab_video:
    st.subheader("Upload Target Video File")
    uploaded = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        
        cap = cv2.VideoCapture(tmp_path)
        col_display, col_metrics = st.columns([2, 1])
        with col_display: 
            video_placeholder = st.empty()
        with col_metrics:
            emotion_metric = st.empty()
            gaze_metric = st.empty()
        
        if st.button("Trigger Computation Node", type="primary"):
            video_history_buffer = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                # 🎯 PROCESS VIDEO FRAME WITH ALL UNET & GAZE OVERLAYS
                processed_frame, gaze, detected, emotion = local_process_frame(frame, video_history_buffer)
                
                emotion_metric.metric(label="🧠 Predicted Emotion", value=str(emotion))
                gaze_metric.code(f"Gaze (X,Y):\n({gaze[0]:.2f}, {gaze[1]:.2f})")
                
                # Display processed frame with segmentation mask and green bounding box
                video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                time.sleep(0.01)
                
            cap.release()
            try: os.unlink(tmp_path)
            except Exception: pass
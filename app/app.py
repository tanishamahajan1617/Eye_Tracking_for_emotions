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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# Streamlit-WebRTC Logs Supress
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

# Drive File IDs
SEG_FILE_ID = "11dayKwl4X3UUfERRpyl6s-nz_YXAZvcA"       
GAZE_FILE_ID = "1EvaC29K0VoCsc7xG72j571cz6mumlsU7"          
EMOTION_FILE_ID = "1Wh4Rro4jkj9_xCoTs1G11ZA5pNVUPMr7"  
SCALER_FILE_ID = "1uOtZmD7900j5hbSfV4WTJ8DVvec7B-0r"

with st.spinner("Syncing Cloud Architecture... Checking Weights & Scaler..."):
    if not WEIGHTS_SEG.exists():
        st.info("Downloading UNet Weights...")
        download_file_from_google_drive(SEG_FILE_ID, WEIGHTS_SEG)
    if not WEIGHTS_GAZE.exists():
        st.info("Downloading Gaze Weights...")
        download_file_from_google_drive(GAZE_FILE_ID, WEIGHTS_GAZE)
    if not WEIGHTS_EMOTION.exists():
        st.info("Downloading Emotion LSTM Weights...")
        download_file_from_google_drive(EMOTION_FILE_ID, WEIGHTS_EMOTION)
    if not SCALER_FILE.exists():
        st.info("Downloading Gaze Scaler...")
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

GAZE_HISTORY = []
SEQUENCE_LENGTH = 30
EMOTION_CLASSES = ["Neutral", "Frustrated", "Bored", "Confident"]

# --- 💻 CORE ANALYTICS ENGINE ---
def local_process_frame(frame):
    global GAZE_HISTORY
    eye_detected = False
    gaze_vectors = [0.5, 0.5]
    pupil_size = 0.33
    detected_emotion = "Neutral"
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w, _ = frame.shape
    
    # Default Center Crop (Smart fallback focusing on face/eye zone)
    ex, ey, ew, eh = int(w * 0.3), int(h * 0.25), int(w * 0.4), int(h * 0.25)

    # 🔒 Safe Cascade Classifier Loading
    eye_cascade = None
    if hasattr(cv2, 'CascadeClassifier'):
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            eye_cascade = cv2.CascadeClassifier(cascade_path)
        except Exception:
            eye_cascade = None

    if eye_cascade is not None and not eye_cascade.empty():
        detected_eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        valid_eyes = []
        for (x, y, bw, bh) in detected_eyes:
            # Filter background detections
            if y < h * 0.6 and x > w * 0.1 and x < w * 0.8:
                valid_eyes.append((x, y, bw, bh))
        
        if len(valid_eyes) > 0:
            ex, ey, ew, eh = valid_eyes[0]
            eye_detected = True

    eye_crop = frame[ey:ey+eh, ex:ex+ew]
    
    # 1. UNet Eye Segmentation
    if seg_model is not None and eye_crop.size > 0:
        try:
            img_t = cv2.resize(eye_crop, (256, 256)).transpose((2, 0, 1)) / 255.0
            img_t = torch.tensor([img_t], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                seg_out = seg_model(img_t)
                pred_mask = torch.sigmoid(seg_out).squeeze().cpu().numpy()
                
                pupil_pixels = np.sum(pred_mask > 0.5)
                pupil_size = float(np.clip(pupil_pixels / (256 * 256), 0.1, 0.8))
                
                iris_mask = cv2.resize((pred_mask > 0.5).astype(np.uint8) * 255, (ew, eh))
                pupil_mask = cv2.resize((pred_mask > 0.5).astype(np.uint8) * 255, (ew, eh))
            
            overlay = frame[ey:ey+eh, ex:ex+ew].copy()
            overlay[iris_mask > 0] = (255, 255, 0)
            overlay[pupil_mask > 0] = (180, 105, 255)
            cv2.addWeighted(overlay, 0.6, frame[ey:ey+eh, ex:ex+ew], 0.4, 0, frame[ey:ey+eh, ex:ex+ew])
            eye_detected = True
        except Exception: 
            pupil_size = 0.33
    
    # 2. Gaze Estimation
    if gaze_model is not None and eye_crop.size > 0:
        try:
            gaze_input = cv2.resize(eye_crop, (64, 64)).transpose((2, 0, 1)) / 255.0
            gaze_input = torch.tensor([gaze_input], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                gaze_out = gaze_model(gaze_input)
                gaze_vectors = gaze_out.squeeze().cpu().tolist()
        except Exception: pass

    # Draw Eye Box
    cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Feature Matrix
    current_features = [float(gaze_vectors[0]), float(gaze_vectors[1]), float(pupil_size)]
    
    GAZE_HISTORY.append(current_features)
    if len(GAZE_HISTORY) > SEQUENCE_LENGTH: 
        GAZE_HISTORY.pop(0)
    
    # 3. Emotion LSTM Inference (Runs continuously once 30 frames collect)
    if emotion_model is not None and len(GAZE_HISTORY) == SEQUENCE_LENGTH:
        try:
            raw_seq = np.array(GAZE_HISTORY, dtype=np.float32)
            scaled_seq = gaze_scaler.transform(raw_seq) if gaze_scaler is not None else raw_seq
            seq_tensor = torch.tensor([scaled_seq], dtype=torch.float32).to(DEVICE)
            
            with torch.no_grad():
                emotion_out = emotion_model(seq_tensor)
                pred_idx = torch.argmax(emotion_out, dim=1).item()
                detected_emotion = EMOTION_CLASSES[pred_idx]
        except Exception: 
            detected_emotion = "Neutral"
        
    return frame, current_features, eye_detected, detected_emotion


# --- 🎥 WEBRTC VIDEO TRANSFORMER WORKER ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class EyeTrackerVideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process live frame through AI pipeline
        processed_frame, gaze, detected, emotion = local_process_frame(img)
        
        # Overlay Live Metrics directly onto video stream
        cv2.putText(
            processed_frame, 
            f"State: {emotion}", 
            (30, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.0, 
            (0, 255, 0), 
            2
        )
        if gaze is not None:
            cv2.putText(
                processed_frame, 
                f"Gaze (X,Y): ({gaze[0]:.2f}, {gaze[1]:.2f})", 
                (30, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                1
            )
            
        return processed_frame


# --- 🖥️ STREAMLIT UI LAYOUT ---
st.set_page_config(page_title="Vision AI Production Node", layout="wide")
st.title("👁️ Enterprise Vision & Emotion Monitor (Cloud Live)")

tab_live, tab_video = st.tabs(["📡 Continuous Live Feed", "🎥 File Video Analyzer"])

with tab_live:
    st.subheader("🔴 Live Stream Eye-Tracking & Emotion Recognition")
    st.write("Click **START** below to launch your camera stream and run real-time inference.")

    webrtc_streamer(
        key="eye-tracker-live-stream",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=EyeTrackerVideoTransformer,
        async_transform=True,
    )

with tab_video:
    st.subheader("Upload Target Video File")
    uploaded = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])
    if uploaded is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        col_display, col_metrics = st.columns(2)
        with col_display: video_placeholder = st.empty()
        with col_metrics:
            emotion_metric = st.empty()
            gaze_metric = st.empty()
        
        if st.button("Trigger Computation Node", type="primary"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                processed_frame, gaze, detected, emotion = local_process_frame(frame)
                emotion_metric.metric(label="🧠 Predicted State", value=str(emotion))
                gaze_metric.code(f"Gaze (X,Y,Pupil):\n{gaze}")
                video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                time.sleep(0.01)
            cap.release()
            try: os.unlink(tmp_path)
            except Exception: pass
import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import time
import torch
import sys
import requests
from pathlib import Path
# WebRTC Imports for Live Webcam on Cloud (Sahi imports)
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# --- 📁 PATHS MANAGEMENT & MODEL IMPORTS ---
ROOT_DIR = Path(__file__).parent.parent  # Root folder tak pahunchne ke liye
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Cloud Paths for Weights
WEIGHTS_SEG = ROOT_DIR / "best_unet_model.pth"
WEIGHTS_GAZE = ROOT_DIR / "best_gaze_model.pth"
WEIGHTS_EMOTION = ROOT_DIR / "best_emotion_lstm.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 📥 GOOGLE DRIVE AUTO-DOWNLOADER ---
def download_file_from_google_drive(file_id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = None
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    if token:
        response = session.get(URL, params={'id': file_id, 'confirm': token}, stream=True)
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: f.write(chunk)

SEG_FILE_ID = "11dayKwl4X3UUfERRpyl6s-nz_YXAZvcA"       
GAZE_FILE_ID = "1EvaC29K0VoCsc7xG72j571cz6mumlsU7"           
EMOTION_FILE_ID = "1Wh4Rro4jkj9_xCoTs1G11ZA5pNVUPMr7"     

# Automatic Download Trigger
with st.spinner("Syncing Cloud Architecture... Checking Weights..."):
    if not WEIGHTS_SEG.exists() and SEG_FILE_ID != "YOUR_UNET_DRIVE_FILE_ID_HERE":
        st.info("Downloading UNet Weights...")
        download_file_from_google_drive(SEG_FILE_ID, WEIGHTS_SEG)
    if not WEIGHTS_GAZE.exists() and GAZE_FILE_ID != "YOUR_GAZE_DRIVE_FILE_ID_HERE":
        st.info("Downloading Gaze Weights...")
        download_file_from_google_drive(GAZE_FILE_ID, WEIGHTS_GAZE)
    if not WEIGHTS_EMOTION.exists() and EMOTION_FILE_ID != "YOUR_EMOTION_DRIVE_FILE_ID_HERE":
        st.info("Downloading Emotion LSTM Weights...")
        download_file_from_google_drive(EMOTION_FILE_ID, WEIGHTS_EMOTION)

# Direct Imports from Models folder
try:
    from Models.eyesegementation_model import UNet
    from Models.gaze_model import GazeModel
    from Models.emotion_model import EmotionLSTM  
    models_imported = True
except ImportError as e:
    models_imported = False
    st.error(f"⚠️ Model classes import error: {e}")

# --- 🚀 CACHE MODELS FOR SPEED ---
@st.cache_resource
def load_vision_models():
    seg, gaze, emotion = None, None, None
    if models_imported:
        if WEIGHTS_SEG.exists():
            seg = UNet().to(DEVICE); seg.eval()
        if WEIGHTS_GAZE.exists():
            gaze = GazeModel().to(DEVICE); gaze.eval()
        if WEIGHTS_EMOTION.exists():
            try:
                emotion = EmotionLSTM().to(DEVICE); emotion.eval()
            except Exception: pass
    return seg, gaze, emotion

seg_model, gaze_model, emotion_model = load_vision_models()

# Global Buffer for Emotion Sequence
GAZE_HISTORY = []
SEQUENCE_LENGTH = 30
EMOTION_CLASSES = ["Neutral", "Focused", "Distracted"]

# --- 💻 STREAMLIT INTERFACE ---
st.set_page_config(page_title="Vision AI Production Node", layout="wide")
st.title("👁️ Enterprise Vision & Emotion Monitor (Cloud Live)")

tab_video, tab_live = st.tabs(["🎥 Network Video Analyzer", "📸 Live Webcam Node"])

def local_process_frame(frame):
    global GAZE_HISTORY
    eye_detected = False
    gaze_vectors = [0.5, 0.5]
    detected_emotion = "Baseline Engine Active"
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(eyes) == 0:
        h, w, _ = frame.shape
        eyes = [[int(w*0.3), int(h*0.3), int(w*0.3), int(w*0.3)]]
    else: eye_detected = True

    for (ex, ey, ew, eh) in eyes:
        eye_crop = frame[ey:ey+eh, ex:ex+ew]
        
        # 1. Segmentation
        if seg_model is not None:
            try:
                img_t = cv2.resize(eye_crop, (256, 256)).transpose((2, 0, 1)) / 255.0
                img_t = torch.tensor([img_t], dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    seg_out = seg_model(img_t)
                    pred_mask = torch.sigmoid(seg_out).squeeze().cpu().numpy()
                    iris_mask = cv2.resize((pred_mask > 0.5).astype(np.uint8) * 255, (ew, eh))
                    pupil_mask = cv2.resize((pred_mask > 0.5).astype(np.uint8) * 255, (ew, eh))
                overlay = frame[ey:ey+eh, ex:ex+ew].copy()
                overlay[iris_mask > 0] = (255, 255, 0)
                overlay[pupil_mask > 0] = (180, 105, 255)
                cv2.addWeighted(overlay, 0.6, frame[ey:ey+eh, ex:ex+ew], 0.4, 0, frame[ey:ey+eh, ex:ex+ew])
            except Exception: pass
        
        # 2. Gaze Estimation
        if gaze_model is not None:
            try:
                gaze_input = cv2.resize(eye_crop, (64, 64)).transpose((2, 0, 1)) / 255.0
                gaze_input = torch.tensor([gaze_input], dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    gaze_out = gaze_model(gaze_input)
                    gaze_vectors = gaze_out.squeeze().cpu().tolist()
            except Exception: pass
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        break

    # 3. Emotion Pipeline Sequence Logger
    GAZE_HISTORY.append(gaze_vectors)
    if len(GAZE_HISTORY) > SEQUENCE_LENGTH: GAZE_HISTORY.pop(0)
    
    if emotion_model is not None and len(GAZE_HISTORY) == SEQUENCE_LENGTH:
        try:
            seq_tensor = torch.tensor([GAZE_HISTORY], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                emotion_out = emotion_model(seq_tensor)
                pred_idx = torch.argmax(emotion_out, dim=1).item()
                detected_emotion = EMOTION_CLASSES[pred_idx]
        except Exception: detected_emotion = "Processing Sequence..."
        
    return frame, gaze_vectors, eye_detected, detected_emotion

# --- 🎥 MODE 1: VIDEO UPLOADER ---
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
        
        if st.button("Trigger Internal Computation Node", type="primary"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                processed_frame, gaze, detected, emotion = local_process_frame(frame)
                emotion_metric.metric(label="🧠 Predicted State", value=str(emotion))
                gaze_metric.code(f"Gaze Vector:\n{gaze}")
                video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                time.sleep(0.01)
            cap.release()
            try: os.unlink(tmp_path)
            except Exception: pass

# --- 📸 MODE 2: LIVE WEBCAM NODE ---
with tab_live:
    st.subheader("📹 Real-time Cloud WebRTC Node")
    st.write("Click 'Start' below to stream your laptop camera to the cloud computing cluster.")

    # Wrapper class jo webcam ke real-time video frames process karegi
    class CloudVideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            # 1. Frame ko WebRTC standard se OpenCV BGR image format mein badlo
            img = frame.to_ndarray(format="bgr24")
            
            # 2. Sahi model pipeline call karo
            try:
                processed_img, gaze, detected, emotion = local_process_frame(img)
                
                # Render state text overlays directly onto the video stream
                cv2.putText(processed_img, f"State: {emotion}", (20, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            except Exception as e:
                processed_img = img

            # 3. Processed frame ko browser par wapas real-time send karo
            return frame.from_ndarray(processed_img, format="bgr24")

    # Fixed: Sahi function name 'webrtc_streamer' call kiya hai yahan
    webrtc_streamer(
        key="cloud-eye-tracking-stream",
        video_processor_factory=CloudVideoProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}  # Google STUN server for firewall bypass
        ),
        media_stream_constraints={"video": True, "audio": False},
    )
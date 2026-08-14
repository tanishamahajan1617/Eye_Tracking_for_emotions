import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import joblib
import time
import gdown
import sys
import os
from pathlib import Path

# --- 1. SAFE MEDIAPIPE FACE MESH IMPORT ---
import mediapipe as mp

try:
    mp_face_mesh = mp.solutions.face_mesh
except AttributeError:
    import importlib
    mp_face_mesh = importlib.import_module('mediapipe.python.solutions.face_mesh')

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)

# --- 2. PAGE CONFIGURATION ---
st.set_page_config(page_title="Eye Tracking & Emotion AI", layout="wide")
st.title("👁️ Dynamic Eye Tracking & Emotion Recognition")
st.caption("Real-time Dynamic Eye Segmentation Overlay (UNet) & Sequence-based Emotion Classification (LSTM)")

# --- 3. PATHS & AUTO-DOWNLOADER ---
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

COLOR_SCLERA = (0, 255, 0)     # 🟢 Green
COLOR_IRIS = (255, 255, 0)     # 🩵 Cyan 
COLOR_PUPIL = (255, 0, 255)    # 🩷 Magenta

def download_file_from_google_drive(file_id, destination):
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, str(destination), quiet=False)

SEG_FILE_ID = "11dayKwl4X3UUfERRpyl6s-nz_YXAZvcA"       
GAZE_FILE_ID = "1EvaC29K0VoCsc7xG72j571cz6mumlsU7"          
EMOTION_FILE_ID = "1Wh4Rro4jkj9_xCoTs1G11ZA5pNVUPMr7"  
SCALER_FILE_ID = "1uOtZmD7900j5hbSfV4WTJ8DVvec7B-0r"

with st.spinner("Downloading/Loading Models & Scaler..."):
    if not WEIGHTS_SEG.exists():
        download_file_from_google_drive(SEG_FILE_ID, WEIGHTS_SEG)
    if not WEIGHTS_GAZE.exists():
        download_file_from_google_drive(GAZE_FILE_ID, WEIGHTS_GAZE)
    if not WEIGHTS_EMOTION.exists():
        download_file_from_google_drive(EMOTION_FILE_ID, WEIGHTS_EMOTION)
    if not SCALER_FILE.exists():
        download_file_from_google_drive(SCALER_FILE_ID, SCALER_FILE)

# --- 4. MODEL LOADERS ---
@st.cache_resource
def load_all_assets():
    try:
        from Models.eyesegementation_model import UNet
        from Models.gaze_model import GazeModel
        from Models.emotion_model import EmotionLSTM
        
        seg_model = UNet().to(DEVICE)
        seg_model.load_state_dict(torch.load(WEIGHTS_SEG, map_location=DEVICE))
        seg_model.eval()

        try:
            expected_in_channels = seg_model.inc.double_conv[0].in_channels
        except Exception:
            expected_in_channels = 3 

        gaze_model = GazeModel().to(DEVICE)
        gaze_model.load_state_dict(torch.load(WEIGHTS_GAZE, map_location=DEVICE))
        gaze_model.eval()

        emotion_model = EmotionLSTM(input_size=3, num_classes=4).to(DEVICE)
        emotion_model.load_state_dict(torch.load(WEIGHTS_EMOTION, map_location=DEVICE), strict=False)
        emotion_model.eval()

        scaler = joblib.load(SCALER_FILE)

        return seg_model, gaze_model, emotion_model, scaler, expected_in_channels, True
    except Exception as e:
        st.error(f"Error loading models or scaler: {e}")
        return None, None, None, None, 3, False

seg_model, gaze_model, emotion_model, gaze_scaler, expected_in_channels, loaded_ok = load_all_assets()

# --- 5. ACCURATE MEDIAPIPE EYE BOUNDING BOX CROPPER ---
def extract_dynamic_eye_region(frame):
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    try:
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Specific landmarks strictly around both eyes (Left & Right Eyes)
            eye_pts_idx = [
                33, 133, 160, 159, 158, 144, 145, 153,  # Left Eye
                362, 263, 387, 386, 385, 373, 374, 380  # Right Eye
            ]
            
            x_coords = [int(landmarks[idx].x * w) for idx in eye_pts_idx]
            y_coords = [int(landmarks[idx].y * h) for idx in eye_pts_idx]

            padding_x = int(w * 0.02)
            padding_y = int(h * 0.015)

            min_x = max(0, min(x_coords) - padding_x)
            max_x = min(w, max(x_coords) + padding_x)
            min_y = max(0, min(y_coords) - padding_y)
            max_y = min(h, max(y_coords) + padding_y)

            ex, ey = min_x, min_y
            ew, eh = max_x - min_x, max_y - min_y
            
            if ew > 20 and eh > 20:
                return ex, ey, ew, eh
    except Exception:
        pass

    # Tighter fallback tightly cropped to eye band only
    return int(w * 0.15), int(h * 0.32), int(w * 0.70), int(h * 0.18)

# --- 6. FRAME PROCESSING FUNCTION ---
def process_frame(frame, sequence_buffer, frame_count, last_emotion):
    h, w, _ = frame.shape
    
    ex, ey, ew, eh = extract_dynamic_eye_region(frame)
    eye_crop = frame[ey:ey+eh, ex:ex+ew]

    gaze_x, gaze_y = 0.5, 0.5
    pupil_ratio = 0.33
    
    if eye_crop is not None and eye_crop.shape[0] > 10 and eye_crop.shape[1] > 10:
        # --- A. UNET EYE SEGMENTATION HIGHLIGHTING ---
        resized_crop = cv2.resize(eye_crop, (256, 256))
        
        if expected_in_channels == 1:
            gray_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2GRAY)
            img_seg_np = np.expand_dims(gray_crop, axis=0).astype(np.float32) / 255.0
        else:
            rgb_crop = cv2.cvtColor(resized_crop, cv2.COLOR_BGR2RGB)
            img_seg_np = rgb_crop.transpose((2, 0, 1)).astype(np.float32) / 255.0
            
        img_seg_t = torch.from_numpy(img_seg_np).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            seg_out = seg_model(img_seg_t)
            
            if seg_out.shape[1] > 1:
                pred_mask = torch.argmax(seg_out, dim=1).squeeze(0).cpu().numpy()
            else:
                pred_mask = (torch.sigmoid(seg_out).squeeze().cpu().numpy() > 0.25).astype(np.uint8)

            mask_resized = cv2.resize(pred_mask.astype(np.uint8), (ew, eh), interpolation=cv2.INTER_NEAREST)

            color_mask = np.zeros_like(eye_crop, dtype=np.uint8)
            if seg_out.shape[1] > 1:
                color_mask[mask_resized == 1] = COLOR_SCLERA  # 🟢 Green
                color_mask[mask_resized == 2] = COLOR_IRIS    # 🩵 Cyan
                color_mask[mask_resized == 3] = COLOR_PUPIL   # 🩷 Magenta
            else:
                color_mask[mask_resized == 1] = COLOR_SCLERA

            overlay = eye_crop.copy()
            has_mask = np.any(color_mask > 0, axis=-1)
            overlay[has_mask] = color_mask[has_mask]
            
            cv2.addWeighted(overlay, 0.6, eye_crop, 0.4, 0, frame[ey:ey+eh, ex:ex+ew])

            pupil_pixels = np.sum(mask_resized == 3) if seg_out.shape[1] > 1 else np.sum(mask_resized == 1)
            pupil_ratio = float(np.clip(pupil_pixels / (ew * eh), 0.05, 0.8))

        # --- B. GAZE PREPROCESSING ---
        rgb_gaze = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2RGB)
        resized_gaze = cv2.resize(rgb_gaze, (64, 64))
        img_gaze_np = resized_gaze.transpose((2, 0, 1)).astype(np.float32) / 255.0
        img_gaze_t = torch.from_numpy(img_gaze_np).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            gaze_out = gaze_model(img_gaze_t)
            gaze_coords = gaze_out.squeeze().cpu().tolist()
            if isinstance(gaze_coords, list) and len(gaze_coords) >= 2:
                gaze_x, gaze_y = float(gaze_coords[0]), float(gaze_coords[1])

        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv2.putText(frame, "Dynamic Eye Zone", (ex, ey - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    sequence_buffer.append([gaze_x, gaze_y, pupil_ratio])
    if len(sequence_buffer) > 30:
        sequence_buffer.pop(0)

    # --- C. EMOTION PREDICTION (EVERY 5 FRAMES) ---
    predicted_emotion = last_emotion
    if len(sequence_buffer) == 30 and (frame_count % 5 == 0):
        raw_seq = np.array(sequence_buffer, dtype=np.float32)
        scaled_seq = gaze_scaler.transform(raw_seq) if gaze_scaler is not None else raw_seq
        seq_tensor = torch.tensor([scaled_seq], dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            emotion_out = emotion_model(seq_tensor)
            pred_idx = torch.argmax(emotion_out, dim=1).item()
            predicted_emotion = EMOTION_CLASSES[pred_idx]

    return frame, gaze_x, gaze_y, predicted_emotion

# --- 7. USER INTERFACE & RENDERED VIDEO PLAYER ---
if loaded_ok:
    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.subheader("📊 Live Pipeline Metrics")
        emotion_metric = st.empty()
        gaze_metric = st.empty()
        
        st.markdown("---")
        st.markdown("### 🎨 UNet Segmentation Palette")
        st.markdown("- 🟢 **Sclera:** Green Highlight")
        st.markdown("- 🩵 **Iris:** Cyan Highlight")
        st.markdown("- 🩷 **Pupil:** Magenta Highlight")

    with col_left:
        st.subheader("🎥 Dynamic AI Video Feed")
        uploaded_video = st.file_uploader("Upload Input Video (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])

    if uploaded_video is not None:
        temp_input = Path("temp_input_video.mp4")
        temp_output = Path("temp_output_video.mp4")
        
        with open(temp_input, "wb") as f:
            f.write(uploaded_video.read())

        if st.button("▶️ Process & Render AI Video", type="primary"):
            cap = cv2.VideoCapture(str(temp_input))
            
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(temp_output), fourcc, fps, (width, height))

            progress_bar = st.progress(0, text="🤖 Processing Frames with MediaPipe & UNet...")
            
            sequence_buffer = []
            frame_count = 0
            current_emotion = "Analyzing..."

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                
                processed_frame, gx, gy, current_emotion = process_frame(
                    frame, sequence_buffer, frame_count, current_emotion
                )

                out.write(processed_frame)
                
                if total_frames > 0:
                    progress_bar.progress(min(frame_count / total_frames, 1.0))

            cap.release()
            out.release()
            progress_bar.empty()

            st.success("✅ AI Processing Complete!")
            
            # Display rendered processed video with full player controls
            with open(temp_output, 'rb') as v_file:
                video_bytes = v_file.read()
                st.video(video_bytes)

            emotion_metric.metric(label="🧠 Final Emotion Prediction", value=current_emotion)

            if temp_input.exists():
                temp_input.unlink()
            if temp_output.exists():
                temp_output.unlink()
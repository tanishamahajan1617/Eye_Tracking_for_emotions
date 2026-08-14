import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
import joblib
import time
from pathlib import Path

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="Eye Tracking & Emotion AI Demo", layout="wide")
st.title("👁️ Eye Tracking & Emotion Recognition System")
st.caption("Real-time Eye Segmentation (UNet), Gaze Estimation, and Sequence-based Emotion Classification (LSTM)")

# --- 2. PATHS & DEVICE SETUP (FLEXIBLE ROOT RESOLUTION) ---
CURRENT_DIR = Path(__file__).parent
REPO_ROOT = CURRENT_DIR.parent

# Helper function to find weights in either root or /app folder
def resolve_path(filename):
    if (REPO_ROOT / filename).exists():
        return REPO_ROOT / filename
    return CURRENT_DIR / filename

WEIGHTS_SEG = resolve_path("best_unet_model.pth")
WEIGHTS_GAZE = resolve_path("best_gaze_model.pth")
WEIGHTS_EMOTION = resolve_path("best_emotion_lstm.pth")
SCALER_FILE = resolve_path("gaze_scaler.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMOTION_CLASSES = ["Neutral", "Frustrated", "Bored", "Confident"]

# --- 3. MODEL ARCHITECTURES (Simplified Inference Loaders) ---
@st.cache_resource
def load_all_assets():
    """Loads and caches all weights and scalers efficiently."""
    try:
        from Models.eyesegementation_model import UNet
        from Models.gaze_model import GazeModel
        from Models.emotion_model import EmotionLSTM
        
        # Load UNet
        seg_model = UNet().to(DEVICE)
        seg_model.load_state_dict(torch.load(WEIGHTS_SEG, map_location=DEVICE))
        seg_model.eval()

        # Load Gaze Model
        gaze_model = GazeModel().to(DEVICE)
        gaze_model.load_state_dict(torch.load(WEIGHTS_GAZE, map_location=DEVICE))
        gaze_model.eval()

        # Load Emotion LSTM
        emotion_model = EmotionLSTM(input_size=3, num_classes=4).to(DEVICE)
        emotion_model.load_state_dict(torch.load(WEIGHTS_EMOTION, map_location=DEVICE))
        emotion_model.eval()

        # Load Scaler
        scaler = joblib.load(SCALER_FILE)

        return seg_model, gaze_model, emotion_model, scaler, True
    except Exception as e:
        st.error(f"Error loading models or scaler: {e}")
        return None, None, None, None, False

seg_model, gaze_model, emotion_model, gaze_scaler, loaded_ok = load_all_assets()

# --- 4. FRAME PROCESSING FUNCTION ---
def process_frame(frame, sequence_buffer):
    """Processes a single video frame through UNet -> Gaze -> Emotion LSTM pipeline."""
    h, w, _ = frame.shape
    
    # Define primary eye zone crop (Center upper region offset)
    ex, ey = int(w * 0.15), int(h * 0.35)
    ew, eh = int(w * 0.70), int(h * 0.40)
    eye_crop = frame[ey:ey+eh, ex:ex+ew]

    gaze_x, gaze_y = 0.5, 0.5
    pupil_ratio = 0.33
    
    if eye_crop.size > 0:
        # A. UNet Segmentation & Highlight Blending
        img_seg = cv2.resize(eye_crop, (256, 256)).transpose((2, 0, 1)) / 255.0
        img_seg_t = torch.tensor([img_seg], dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            seg_out = seg_model(img_seg_t)
            
            if seg_out.shape[1] > 1:
                pred_mask = torch.argmax(seg_out, dim=1).squeeze().cpu().numpy()
            else:
                pred_mask = (torch.sigmoid(seg_out).squeeze().cpu().numpy() > 0.25).astype(np.uint8)

            mask_resized = cv2.resize(pred_mask.astype(np.uint8), (ew, eh), interpolation=cv2.INTER_NEAREST)

            # Build Multi-Color Overlay (Sclera: Green, Iris: Cyan, Pupil: Magenta)
            color_mask = np.zeros_like(eye_crop, dtype=np.uint8)
            if seg_out.shape[1] > 1:
                color_mask[mask_resized == 1] = [0, 255, 0]    # 🟢 Sclera
                color_mask[mask_resized == 2] = [255, 255, 0]  # 🩵 Iris
                color_mask[mask_resized == 3] = [255, 0, 255]  # 🩷 Pupil
            else:
                color_mask[mask_resized == 1] = [0, 255, 0]

            # Alpha Blend Mask onto Frame
            overlay = eye_crop.copy()
            has_mask = np.any(color_mask > 0, axis=-1)
            overlay[has_mask] = color_mask[has_mask]
            cv2.addWeighted(overlay, 0.7, eye_crop, 0.3, 0, frame[ey:ey+eh, ex:ex+ew])

            # Pupil feature extraction
            pupil_pixels = np.sum(mask_resized == 3) if seg_out.shape[1] > 1 else np.sum(mask_resized == 1)
            pupil_ratio = float(np.clip(pupil_pixels / (ew * eh), 0.05, 0.8))

        # B. Gaze Estimation
        img_gaze = cv2.resize(eye_crop, (64, 64)).transpose((2, 0, 1)) / 255.0
        img_gaze_t = torch.tensor([img_gaze], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            gaze_out = gaze_model(img_gaze_t)
            gaze_coords = gaze_out.squeeze().cpu().tolist()
            gaze_x, gaze_y = float(gaze_coords[0]), float(gaze_coords[1])

        # Draw Eye Zone Box
        cv2.rectangle(frame, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        cv2.putText(frame, "Eye Segmented Zone", (ex, ey - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # C. Sequence Buffering & Emotion LSTM Prediction
    current_step = [gaze_x, gaze_y, pupil_ratio]
    sequence_buffer.append(current_step)
    if len(sequence_buffer) > 30:
        sequence_buffer.pop(0)

    predicted_emotion = "Gathering Frames..."
    if len(sequence_buffer) == 30:
        raw_seq = np.array(sequence_buffer, dtype=np.float32)
        scaled_seq = gaze_scaler.transform(raw_seq)
        seq_tensor = torch.tensor([scaled_seq], dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            emotion_out = emotion_model(seq_tensor)
            pred_idx = torch.argmax(emotion_out, dim=1).item()
            predicted_emotion = EMOTION_CLASSES[pred_idx]

    return frame, gaze_x, gaze_y, predicted_emotion


# --- 5. DEMO USER INTERFACE ---
if loaded_ok:
    col_left, col_right = st.columns([2, 1])

    with col_right:
        st.subheader("📊 Live Pipeline Metrics")
        emotion_metric = st.empty()
        gaze_metric = st.empty()
        
        st.markdown("---")
        st.markdown("### 🎨 UNet Color Mask Key")
        st.markdown("- 🟢 **Sclera:** Green")
        st.markdown("- 🩵 **Iris:** Cyan")
        st.markdown("- 🩷 **Pupil:** Magenta")

    with col_left:
        st.subheader("🎥 Demonstration Feed")
        uploaded_video = st.file_uploader("Upload Demo Video File (.mp4, .avi, .mov)", type=["mp4", "avi", "mov"])
        video_placeholder = st.empty()

    if uploaded_video is not None:
        # Save temp file for OpenCV reading
        temp_path = Path("temp_demo_video.mp4")
        with open(temp_path, "wb") as f:
            f.write(uploaded_video.read())

        cap = cv2.VideoCapture(str(temp_path))
        sequence_buffer = []

        if st.button("▶️ Run Model Pipeline Demo", type="primary"):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 480))
                processed_frame, gx, gy, emotion = process_frame(frame, sequence_buffer)

                # Update UI
                video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                emotion_metric.metric(label="🧠 Predicted Emotion", value=emotion)
                gaze_metric.code(f"Gaze Vector (X, Y):\n({gx:.2f}, {gy:.2f})")

                time.sleep(0.01) # Smooth playback

            cap.release()
            if temp_path.exists():
                temp_path.unlink()
else:
    st.warning("Please ensure model files (`.pth`) and `gaze_scaler.pkl` exist in the repository or app folder.")
from fastapi import FastAPI, APIRouter, HTTPException
from starlette.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import base64
import sys
from pathlib import Path
from pydantic import BaseModel
from typing import List

# --- 📁 PATHS MANAGEMENT ---
SERVER_DIR = Path(__file__).parent  # app/backend
ROOT_DIR = SERVER_DIR.parent.parent # Root folder

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Dynamic imports for all 3 models
try:
    from Models.eyesegementation_model import UNet
    from Models.gaze_model import GazeModel
    # 👇 CHANGE HERE: Apni actual emotion file aur class ka naam yahan likho
    from Models.emotion_model import EmotionLSTM 
    print("✅ All model classes (UNet, GazeModel, EmotionLSTM) imported successfully!")
except ImportError as e:
    print(f"⚠️ Class Import Warning/Error: {e}")
    UNet = None
    GazeModel = None
    EmotionLSTM = None

# Strict weights mapping
WEIGHTS_SEG = Path(r"C:\Projects\eye_tracking\Eye_Tracking_for_emotions\best_unet_model.pth")
WEIGHTS_GAZE = Path(r"C:\Projects\eye_tracking\Eye_Tracking_for_emotions\best_gaze_model.pth")
WEIGHTS_EMOTION = Path(r"C:\Projects\eye_tracking\Eye_Tracking_for_emotions\best_emotion_lstm.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seg_model, gaze_model, emotion_model = None, None, None
status_msg = "Model loading not initialized"

# LSTM Time-series Buffer Configuration
GAZE_HISTORY: List[List[float]] = []
SEQUENCE_LENGTH = 30  # Timesteps for LSTM
EMOTION_CLASSES = ["Neutral", "Focused", "Distracted"] # Apni classes ke mutabik change karlo

# --- 🧠 LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global seg_model, gaze_model, emotion_model, status_msg
    msgs = []
    
    # 1. UNet Load
    if UNet and WEIGHTS_SEG.exists():
        try:
            seg_model = UNet().to(DEVICE)
            seg_model.load_state_dict(torch.load(WEIGHTS_SEG, map_location=DEVICE), strict=False)
            seg_model.eval()
            msgs.append("UNet Loaded")
        except Exception as e: msgs.append(f"UNet Error: {str(e)}")
    else: msgs.append("UNet missing")

    # 2. Gaze Load
    if GazeModel and WEIGHTS_GAZE.exists():
        try:
            gaze_model = GazeModel().to(DEVICE)
            gaze_model.load_state_dict(torch.load(WEIGHTS_GAZE, map_location=DEVICE), strict=False)
            gaze_model.eval()
            msgs.append("Gaze Loaded")
        except Exception as e: msgs.append(f"Gaze Error: {str(e)}")
    else: msgs.append("Gaze missing")

    # 3. Emotion LSTM Load
    if EmotionLSTM and WEIGHTS_EMOTION.exists():
        try:
            # Note: Model architecture dimensions template logic config inside your class file
            emotion_model = EmotionLSTM().to(DEVICE)
            emotion_model.load_state_dict(torch.load(WEIGHTS_EMOTION, map_location=DEVICE), strict=False)
            emotion_model.eval()
            msgs.append("Emotion LSTM Loaded")
        except Exception as e: msgs.append(f"Emotion Error: {str(e)}")
    else: msgs.append("Emotion weights missing")
        
    status_msg = " | ".join(msgs)
    print(f"🎯 Server Startup Diagnostics: {status_msg}")
    yield

# --- 🚀 FASTAPI INSTANCE ---
app = FastAPI(title="Vision AI Inference Server", lifespan=lifespan)
api_router = APIRouter(prefix="/api")

class ProcessFrameRequest(BaseModel):
    image_base64: str

class ProcessFrameResponse(BaseModel):
    processed_image_base64: str
    gaze_coords: List[float]
    eye_detected: bool
    predicted_emotion: str
    server_status: str

@api_router.post("/process-frame", response_model=ProcessFrameResponse)
async def process_frame(req: ProcessFrameRequest):
    global GAZE_HISTORY
    try:
        img_bytes = base64.b64decode(req.image_base64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        eye_detected = False
        gaze_vectors = [0.5, 0.5] # Default center coordinates
        detected_emotion = "Collecting Sequences..."
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(eyes) == 0:
            h, w, _ = frame.shape
            eyes = [[int(w*0.3), int(h*0.3), int(w*0.3), int(w*0.3)]]
        else:
            eye_detected = True

        for (ex, ey, ew, eh) in eyes:
            eye_crop = frame[ey:ey+eh, ex:ex+ew]
            
            # --- Segmentation Pipeline (UNet) ---
            if seg_model is not None:
                try:
                    img_t = cv2.resize(eye_crop, (256, 256)).transpose((2, 0, 1)) / 255.0
                    img_t = torch.tensor([img_t], dtype=torch.float32).to(DEVICE)
                    with torch.no_grad():
                        seg_out = seg_model(img_t)
                        pred_mask = torch.sigmoid(seg_out).squeeze().cpu().numpy()
                        
                        if len(pred_mask.shape) == 3:
                            iris_mask = cv2.resize((pred_mask > 0.5).astype(np.uint8) * 255, (ew, eh))
                            pupil_mask = cv2.resize((pred_mask > 0.5).astype(np.uint8) * 255, (ew, eh))
                        else:
                            iris_mask = cv2.resize((pred_mask > 0.5).astype(np.uint8) * 255, (ew, eh))
                            pupil_mask = cv2.resize((pred_mask > 0.7).astype(np.uint8) * 255, (ew, eh))

                    overlay = frame[ey:ey+eh, ex:ex+ew].copy()
                    overlay[iris_mask > 0] = (255, 255, 0)     # Cyan Iris
                    overlay[pupil_mask > 0] = (180, 105, 255)  # Pink Pupil
                    cv2.addWeighted(overlay, 0.6, frame[ey:ey+eh, ex:ex+ew], 0.4, 0, frame[ey:ey+eh, ex:ex+ew])
                except Exception: pass
            
            # --- Gaze Estimation Pipeline ---
            if gaze_model is not None:
                try:
                    gaze_input = cv2.resize(eye_crop, (64, 64)).transpose((2, 0, 1)) / 255.0
                    gaze_input = torch.tensor([gaze_input], dtype=torch.float32).to(DEVICE)
                    with torch.no_grad():
                        gaze_out = gaze_model(gaze_input)
                        gaze_vectors = gaze_out.squeeze().cpu().tolist() # Returns [x, y]
                except Exception: pass
                
            cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            break

        # --- 📈 EMOTION LSTM ENGINE (Sequence Processor) ---
        GAZE_HISTORY.append(gaze_vectors)
        if len(GAZE_HISTORY) > SEQUENCE_LENGTH:
            GAZE_HISTORY.pop(0) # Maintain window scale sliding size
            
        if emotion_model is not None and len(GAZE_HISTORY) == SEQUENCE_LENGTH:
            try:
                seq_tensor = torch.tensor([GAZE_HISTORY], dtype=torch.float32).to(DEVICE) # Shape:
                with torch.no_grad():
                    emotion_out = emotion_model(seq_tensor)
                    pred_idx = torch.argmax(emotion_out, dim=1).item()
                    detected_emotion = EMOTION_CLASSES[pred_idx]
            except Exception as e:
                detected_emotion = f"LSTM Inference Error"
        elif emotion_model is None:
            detected_emotion = "LSTM Engine Offline"

        _, buffer = cv2.imencode('.jpg', frame)
        processed_b64 = base64.b64encode(buffer).decode('ascii')
        
        return ProcessFrameResponse(
            processed_image_base64=processed_b64,
            gaze_coords=gaze_vectors,
            eye_detected=eye_detected,
            predicted_emotion=detected_emotion,
            server_status=status_msg
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

app.include_router(api_router)
app.add_middleware(CORSMiddleware, allow_credentials=True, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
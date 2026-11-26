# 👁️ Dynamic Eye Tracking & Emotion Recognition System

A real-time Computer Vision and Deep Learning pipeline built with **PyTorch**, **MediaPipe**, **OpenCV**, and **Streamlit**. This system dynamically localizes human eyes from video streams, performs pixel-level anatomical segmentation, predicts continuous gaze vectors, and classifies temporal emotional states in real time.

---

## 🚀 Key Features

- **Dynamic Eye Localization:** Uses MediaPipe Face Mesh (468 3D landmarks) to adaptively crop left and right eye regions with dynamic padding and mathematical occlusion fallbacks.
- **Pixel-Level Eye Segmentation (UNet):** Highlights anatomical eye structures in real time using a 60% alpha-blended color overlay:
  - 🟢 **Sclera:** Green `(0, 255, 0)`
  - 🩵 **Iris:** Cyan `(255, 255, 0)`
  - 🩷 **Pupil:** Magenta `(255, 0, 255)`
- **Lightweight Gaze Estimation (MobileNet CNN):** Extracts continuous $(X, Y)$ gaze direction vectors at high FPS using Depthwise Separable Convolutions.
- **Temporal Emotion Classification (Stacked LSTM):** Evaluates a rolling 30-frame window of $[Gaze\_X, Gaze\_Y, Pupil\_Ratio]$ features scaled via `StandardScaler` to predict **Neutral**, **Frustrated**, **Bored**, or **Confident**.
- **Cloud-Ready Live Streaming:** Streams processed RGB frames directly to the browser via Streamlit's `st.empty()` buffer, eliminating headless Linux FFmpeg hardware codec crashes (`h264_v4l2m2m`).

---

## 🏗️ System Architecture

```text
[ Input Video Frame ]
         │
         ▼
[ Stage 1: MediaPipe Face Mesh ] ──> 16 Eye Landmarks & Dynamic ROI Crop
         │
         ├───────────────────────────────┬───────────────────────────────┐
         ▼                               ▼                               ▼
[ Stage 2A: UNet Segmentation ]  [ Stage 2B: MobileNet Gaze ]    [ Stage 2C: Pupil Extractor ]
 • Input: 256×256 RGB             • Input: 64×64 RGB              • Counts Magenta Pixels
 • Output: 4-Class Pixel Mask     • Output: (X, Y) Coordinates    • Output: Pupil-to-Eye Ratio
         │                               │                               │
         └───────────────────────────────┴───────────────────────────────┘
                                         │
                                         ▼
                            [ Stage 3: Sliding Buffer ]
                             • 30-Frame Rolling Window [1, 30, 3]
                             • StandardScaler Normalization
                                         │
                                         ▼ (Every 5th Frame)
                             [ Stage 4: EmotionLSTM ]
                             • Neutral / Frustrated / Bored / Confident
                                         │
                                         ▼
                       [ Streamlit Live RGB Frame Streamer ]

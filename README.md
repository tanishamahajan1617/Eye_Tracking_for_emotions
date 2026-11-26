👁️ Dynamic Eye Tracking & Emotion Recognition SystemA real-time Computer Vision and Deep Learning pipeline that dynamically localizes ocular regions, performs pixel-level eye segmentation, tracks continuous gaze vectors, and classifies temporal emotional states from video streams.🚀 Key FeaturesAdaptive Eye Localization: Uses MediaPipe 3D Face Mesh (16 targeted ocular contour landmarks) with dynamic margin padding and a mathematical fallback cropper to handle head tilts, scale shifts, and brief occlusions without crashing.Pixel-Level Anatomical Segmentation: A UNet model segments the Sclera, Iris, and Pupil in real time with a 60% alpha-blended visual overlay.Low-Latency Gaze Estimation: A lightweight MobileNet CNN backbone utilizing Depthwise Separable Convolutions predicts normalized $(X, Y)$ gaze coordinates at high frame rates.Temporal Emotion Classification: A Stacked LSTM processes a rolling 30-frame (1-second) sequence of [Gaze_X, Gaze_Y, Pupil_Ratio] scaled via StandardScaler to classify emotions (Neutral, Frustrated, Bored, Confident).Headless Cloud-Ready Streaming: Streams processed RGB OpenCV frames directly to Streamlit's UI buffer, bypassing Linux hardware video codec bottlenecks (h264_v4l2m2m / FFmpeg issues).🏗️ System Architecture & Pipeline[ Input Video Frame ]
         │
         ▼
[ Stage 1: MediaPipe Face Mesh ]  ──> Extracts 16 Eye Landmarks & Dynamic Cropping
         │
         ├───────────────────────────────┬───────────────────────────────┐
         ▼                               ▼                               ▼
[ Stage 2A: UNet Segmentation ]  [ Stage 2B: MobileNet Gaze ]    [ Stage 2C: Pupil Extractor ]
  • Input: 256×256 RGB             • Input: 64×64 RGB              • Counts Magenta Pixels
  • Output: 4-Class Mask           • Output: (X, Y) Coordinates    • Output: Pupil-to-Eye Ratio
         │                               │                               │
         └───────────────────────────────┴───────────────────────────────┘
                                         │
                                         ▼
                            [ Stage 3: Sliding Buffer ]
                             • 30-Frame Rolling Sequence (30×3)
                             • StandardScaler Normalization
                                         │
                                         ▼ (Evaluated Every 5th Frame)
                             [ Stage 4: EmotionLSTM ]
                             • Neutral / Frustrated / Bored / Confident
                                         │
                                         ▼
                       [ Streamlit Live RGB Frame Streamer ]
🧠 Model SpecificationsComponentArchitectureDatasetInput ShapeLoss FunctionOutputEye SegmentationUNet (Encoder-Decoder + Skip Connections)OpenEDS / MVTec[1, 3, 256, 256]Dice + Cross-Entropy4-Class Pixel Mask (Bg, Sclera, Iris, Pupil)Gaze EstimationMobileNet (Depthwise Separable CNN)MPIIGaze / ETH-XGaze[1, 3, 64, 64]Mean Squared Error (MSE)Continuous (Gaze_X, Gaze_Y) $\in [0.0, 1.0]$Emotion ClassifierStacked LSTM + Dense LayerCK+ / Custom Sequences[1, 30, 3]Categorical Cross-Entropy4 Classes (Neutral, Frustrated, Bored, Confident)🎨 UNet Segmentation Color Palette🟢 Sclera (White Outer Layer): Green Highlight (0, 255, 0)🩵 Iris (Pigmented Ring): Cyan Highlight (255, 255, 0)🩷 Pupil (Center Aperture): Magenta Highlight (255, 0, 255)📂 Repository Structure├── Models/
│   ├── eyesegementation_model.py   # UNet architecture definition
│   ├── gaze_model.py               # MobileNet gaze regression architecture
│   └── emotion_model.py            # Stacked LSTM sequence classifier
├── app/
│   ├── app.py                      # Main Streamlit real-time inference & streaming UI
│   ├── best_unet_model.pth         # Auto-downloaded UNet weights
│   ├── best_gaze_model.pth         # Auto-downloaded MobileNet gaze weights
│   ├── best_emotion_lstm.pth       # Auto-downloaded EmotionLSTM weights
│   └── gaze_scaler.pkl             # Pre-fitted StandardScaler for sequence features
├── requirements.txt                # Project dependencies
└── README.md                       # Project documentation
⚙️ Installation & Local SetupClone the repository:git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Create and activate a virtual environment (recommended):python -m venv venv
source venv/bin/activate        # On Linux/macOS
venv\Scripts\activate           # On Windows
Install dependencies:pip install -r requirements.txt
(Ensure opencv-python-headless, mediapipe, torch, torchvision, streamlit, scikit-learn, joblib, and gdown are included in requirements.txt.)Run the Streamlit application:streamlit run app/app.py
Note: On the first run, the app automatically downloads the pre-trained .pth model weights and gaze_scaler.pkl via gdown.⚡ Performance & Engineering HighlightsLatency: ~18–25 ms per frame (22–28 FPS on standard cloud CPUs; 35–40 FPS on CUDA GPUs).Memory Footprint: < 1.2 GB RAM usage, optimized for free-tier cloud deployment.Inference Optimization: The sliding sequence buffer updates continuously every frame, while EmotionLSTM executes every 5th frame to reduce CPU overhead by ~80% without sacrificing temporal accuracy.🔮 Future ImprovementsMulti-face tracking support for group engagement analytics.Integration of Eye Aspect Ratio (EAR) for real-time blink rate and drowsiness detection.ONNX / TensorRT quantization for edge hardware deployment.

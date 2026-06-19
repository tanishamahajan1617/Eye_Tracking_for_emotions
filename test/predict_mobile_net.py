import sys
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 1. Path Fix
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Models.eyesegementation_model import UNet
from Models.gaze_model import GazeModel as MobileNetGazeRegressor


UNET_WEIGHTS = "best_unet_model.pth"
GAZE_WEIGHTS = "best_gaze_model.pth"
IMAGE_PATH = r"C:\Users\LENOVO\OneDrive\Pictures\Screenshots\Screenshot 2026-06-13 211012.png"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_segmented_gaze_pipeline():
    print(f"Running Integrated Pipeline on: {DEVICE}")

    # ==========================================
    # STEP 1: BOTH MODELS INITIALIZE & LOAD
    # ==========================================
    unet_model = UNet(in_channels=1).to(DEVICE)
    unet_model.load_state_dict(torch.load(UNET_WEIGHTS, map_location=DEVICE))
    unet_model.eval()

    gaze_model = MobileNetGazeRegressor().to(DEVICE)
    if os.path.exists(GAZE_WEIGHTS):
        gaze_model.load_state_dict(torch.load(GAZE_WEIGHTS, map_location=DEVICE))
    gaze_model.eval()
    print("Both U-Net and MobileNetV2 loaded successfully!")

    # STEP 2: RAW IMAGE LOADING & GRAYSCALE
    
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        print(f"ERROR: Image not found at {IMAGE_PATH}!")
        return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    orig_h, orig_w = image_gray.shape[:2]

    
    # STEP 3: RUN U-NET TO GET ALL HIGHLIGHTED SEGMENTS

    unet_input = np.expand_dims(image_gray, axis=-1)
    unet_transform = A.Compose([
        A.Resize(height=256, width=256),
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    unet_tensor = unet_transform(image=unet_input)["image"].unsqueeze(0).to(DEVICE).float()

    with torch.no_grad():
        unet_output = unet_model(unet_tensor)
        pred_mask = torch.argmax(unet_output, dim=1).squeeze(0).cpu().numpy()

    
    pred_mask_resized = cv2.resize(pred_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    # Multi-class highlight color grid setup (Sclera=Green, Iris=Cyan, Pupil=Red)
    colors = [
        (0, 255, 0),   # Sclera: Green
        (0, 255, 255), # Iris: Cyan
        (0, 0, 255),   # Pupil: Red
    ]
    colored_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for class_idx, color in enumerate(colors):
        colored_mask[pred_mask_resized == class_idx] = color

    
    overlay_display = cv2.addWeighted(image_rgb, 0.6, colored_mask, 0.4, 0)

    # ==========================================
    # STEP 4: FEED RGB IMAGE TO MOBILENETV2 (EXPECTS 3 CHANNELS)
    # ==========================================
   
    gaze_input = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)

    gaze_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    gaze_tensor = gaze_transform(image=gaze_input)["image"].unsqueeze(0).to(DEVICE).float()

    with torch.no_grad():
        gaze_output = gaze_model(gaze_tensor)
        coords = gaze_output.cpu().numpy().flatten()

    if coords.size >= 2:
        pixel_x = int(coords[0] * orig_w)
        pixel_y = int(coords[1] * orig_h)
    elif coords.size == 1:
        pixel_x = int(coords[0] * orig_w)
        pixel_y = int(coords[0] * orig_h)
    else:
        raise ValueError(f"Unexpected gaze model output shape: {coords.shape}")

    pixel_x = max(0, min(orig_w - 1, pixel_x))
    pixel_y = max(0, min(orig_h - 1, pixel_y))
    print(f"\nMapped Coordinates from Trained Domain: X={pixel_x}, Y={pixel_y}")

    # ==========================================
    # STEP 5: VISUALIZATION & PLOT
    # ==========================================
    final_display = overlay_display.copy()
    cv2.circle(final_display, (pixel_x, pixel_y), radius=12, color=(255, 0, 0), thickness=-1)
    cv2.circle(final_display, (pixel_x, pixel_y), radius=22, color=(255, 255, 255), thickness=3)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("1. Original Raw Input (Fed to MobileNet)")
    plt.imshow(image_rgb)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("2. Highlighted Segments (Sclera/Iris/Pupil)")
    plt.imshow(colored_mask)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title(f"3. Gaze Prediction\n(X: {pixel_x}, Y: {pixel_y})")
    plt.imshow(final_display)
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_segmented_gaze_pipeline()

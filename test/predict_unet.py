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

# ---- CONFIGURATION ----
MODEL_PATH = "best_unet_model.pth"   
IMAGE_PATH = r"C:\Users\LENOVO\OneDrive\Pictures\Screenshots\Screenshot 2026-06-13 211012.png"        
IMAGE_SIZE = 256                    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def predict_and_show():
    print(f"Using Device: {DEVICE}")
    
    # 2. Model Initialize (Strictly 1 Channel as verified by your log)
    model = UNet(in_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded successfully with 1 Input Channel.")

    # 3. Image Load (Grayscale mein reading aur visual copy setup)
    image_bgr = cv2.imread(IMAGE_PATH)
    if image_bgr is None:
        print(f"ERROR: '{IMAGE_PATH}' nahi mili!")
        return
        
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    original_height, original_width = image_gray.shape[:2]

    # Albumentations ko (H, W, 1) input chahiye
    image_input = np.expand_dims(image_gray, axis=-1)

  # 4. Preprocessing (EXACT MATCH WITH YOUR VAL_TRANSFORM)
    transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    augmented = transform(image=image_input)
    # Model ko tensor .float() banakar hi bhejenge jaisa training mein tha
    input_tensor = augmented["image"].unsqueeze(0).to(DEVICE).float()

    # 5. Prediction
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    print(f"Detected Class IDs in this run: {np.unique(pred_mask)}")

    # 6. Resize Mask back to original size
    pred_mask_resized = cv2.resize(pred_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    # 7. Color Setup for Highlights
    # 0: Background (Black), 1: Sclera (Green), 2: Iris (Cyan), 3: Pupil (Red)
    colors = [
        [0, 0, 0],         # 0 -> Background
        [0, 255, 0],       # 1 -> Sclera
        [255, 255, 0],     # 2 -> Iris
        [255, 0, 0]        # 3 -> Pupil
    ]
    
    colored_mask = np.zeros((original_height, original_width, 3), dtype=np.uint8)
    for class_idx, color in enumerate(colors):
        colored_mask[pred_mask_resized == class_idx] = color

    # Transparency Overlay on the original RGB display
    overlay = cv2.addWeighted(image_rgb, 0.6, colored_mask, 0.4, 0)

    # 8. Plot Result Windows
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("1.Original Image")
    plt.imshow(image_rgb)
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.title("2. Highlighted Segments (Mask)")
    plt.imshow(colored_mask)
    plt.axis("off")
    
    plt.subplot(1, 3, 3)
    plt.title("3. Final Overlay")
    plt.imshow(overlay)
    plt.axis("off")
    
    plt.tight_layout()
    print(" Showing perfect window...")
    plt.show()

if __name__ == "__main__":
    predict_and_show()
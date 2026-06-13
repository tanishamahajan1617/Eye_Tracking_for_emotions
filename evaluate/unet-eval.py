import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.eyesegementation_model import UNet
from custom_datasets.openeds import OpenEDSDataset, val_transform

def evaluate_metrics(loader, model, device, num_classes=4):
    model.eval()
    
    # 
    total_intersection = torch.zeros(num_classes).to(device)
    total_union = torch.zeros(num_classes).to(device)
    
    # Dice score ke liye variables
    total_dice_numerator = torch.zeros(num_classes).to(device)
    total_dice_denominator = torch.zeros(num_classes).to(device)

    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Evaluating Test Set"):
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            preds = torch.argmax(outputs, dim=1) # (Batch, H, W)

            # Har class ke liye pixels calculate karna
            for cls in range(num_classes):
                pred_cls = (preds == cls)
                target_cls = (targets == cls)
                
                # IoU calculations
                intersection = (pred_cls & target_cls).float().sum()
                union = (pred_cls | target_cls).float().sum()
                
                total_intersection[cls] += intersection
                total_union[cls] += union
                
                # Dice calculations
                total_dice_numerator[cls] += 2.0 * intersection
                total_dice_denominator[cls] += (pred_cls.float().sum() + target_cls.float().sum())

    # Final Class-wise Metrics calculation
    class_names = ["Background", "Sclera", "Iris", "Pupil"]
    print("\n================ 📊 FINAL TEST METRICS ================")
    
    iou_list = []
    dice_list = []
    
    for cls in range(num_classes):
        # IoU check for division by zero
        if total_union[cls] == 0:
            iou = 0.0
        else:
            iou = (total_intersection[cls] / total_union[cls]).item()
            
        # Dice check for division by zero
        if total_dice_denominator[cls] == 0:
            dice = 0.0
        else:
            dice = (total_dice_numerator[cls] / total_dice_denominator[cls]).item()
            
        iou_list.append(iou)
        dice_list.append(dice)
        
        print(f"Class [{class_names[cls]}]: IoU = {iou:.4f} | Dice (F1) = {dice:.4f}")
        
    print("-------------------------------------------------------")
    print(f"Mean IoU (mIoU)    : {np.mean(iou_list):.4f}")
    print(f"Mean Dice Coefficient: {np.mean(dice_list):.4f}")
    print("=======================================================\n")

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    
    IMAGE_DIR = os.path.join("data", "openeds", "images")
    MASK_DIR = os.path.join("data", "openeds", "labels")
    MODEL_PATH = "best_unet_model.pth" # Aapki saved weights file
    
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file '{MODEL_PATH}' nahi mili! Pehle training poori hone dein.")
        return

    # Data Loading (Wahi same logic jo training mein test set nikalne ke liye tha)
    images_available = set([f[:-4] for f in os.listdir(IMAGE_DIR) if f.lower().endswith('.png')])
    masks_available = set([f[:-4] for f in os.listdir(MASK_DIR) if f.lower().endswith('.npy')])
    all_images = sorted(list(images_available.intersection(masks_available)))
    
    # Test set ko alag nikalna (Same ratio: 80-10-10)
    _, temp_imgs = train_test_split(all_images, test_size=0.2, shuffle=False)
    _, test_imgs = train_test_split(temp_imgs, test_size=0.5, shuffle=False)
    
    print(f"[INFO] Loaded {len(test_imgs)} images for testing.")

    test_ds = OpenEDSDataset(IMAGE_DIR, MASK_DIR, test_imgs, transform=val_transform)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Model Initialize aur Weights Load karna
    model = UNet(in_channels=1, num_classes=4).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print(f"[INFO] Model weights successfully loaded from '{MODEL_PATH}'")

    # Metrics Calculate karna
    evaluate_metrics(test_loader, model, DEVICE, num_classes=4)

if __name__ == "__main__":
    main()
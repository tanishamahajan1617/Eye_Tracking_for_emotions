import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Root path add karna taaki Models aur custom_datasets import ho sakein
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Models.eyesegementation_model import UNet
from custom_datasets.openeds import OpenEDSDataset, train_transform, val_transform

def train_fn(loader, model, optimizer, loss_fn, device):
    model.train()
    loop = tqdm(loader, desc="Training")
    total_loss = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        # 1. Data and Targets send to cuda device
        data = data.to(device)
        targets = targets.to(device)
        
        # ---  check ---
        if batch_idx == 0:
            print(f"\n[CHECK] Model device: {next(model.parameters()).device}")
            print(f"[CHECK] Data device: {data.device}") 
        # ------------------------------

        # Forward pass   
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())
        
    print(f"Average Train Loss: {total_loss / len(loader):.4f}")
def validate_fn(loader, model, loss_fn, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, targets in loader:
            # FIX HERE ALSO: Overwrite variables
            data = data.to(device)
            targets = targets.to(device)
            
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()
            
    avg_val_loss = total_loss / len(loader)
    print(f"Average Validation Loss: {avg_val_loss:.4f}")
    return avg_val_loss
def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 16
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    
    IMAGE_DIR = os.path.join("data", "openeds", "images")
    MASK_DIR = os.path.join("data", "openeds", "labels")
    

    images_available = set()
    for file in os.listdir(IMAGE_DIR):
        if file.lower().endswith('.png'):
            base_name = file[:-4]  # .png (4 characters)
            images_available.add(base_name)
            
    masks_available = set()
    for file in os.listdir(MASK_DIR):
        if file.lower().endswith('.npy'):
            base_name = file[:-4]  # .npy (4 characters)
            masks_available.add(base_name)
    
    # intersection of both sets
    all_images = sorted(list(images_available.intersection(masks_available)))
    
    if len(all_images) == 0:
        print("\n==== CRITICAL MATCH FAILURE DEBUGGER ====")
        print(f"Raw Images Sample: {sorted(os.listdir(IMAGE_DIR))[:3]}")
        print(f"Raw Masks Sample:  {sorted(os.listdir(MASK_DIR))[:3]}")
        print("============================================\n")
        print("[CRITICAL ERROR] Base names match not found!")
        return
        
    print(f" Successfully matched pairs found: {len(all_images)}")

    # Data Splitting
    train_imgs, temp_imgs = train_test_split(all_images, test_size=0.2, shuffle=False)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, shuffle=False)
    
    print(f"Train Size: {len(train_imgs)} | Val Size: {len(val_imgs)} | Test Size: {len(test_imgs)}")

    train_ds = OpenEDSDataset(IMAGE_DIR, MASK_DIR, train_imgs, transform=train_transform)
    val_ds = OpenEDSDataset(IMAGE_DIR, MASK_DIR, val_imgs, transform=val_transform)
    test_ds = OpenEDSDataset(IMAGE_DIR, MASK_DIR, test_imgs, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    model = UNet(in_channels=1, num_classes=4).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_loss = float("inf")
    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} ===")
        train_fn(train_loader, model, optimizer, loss_fn, DEVICE)
        val_loss = validate_fn(val_loader, model, loss_fn, DEVICE)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_unet_model.pth")
            print("=> Saved new best model checkpoint!")

    print("\n=== Final Test Evaluation ===")
    if os.path.exists("best_unet_model.pth"):
        model.load_state_dict(torch.load("best_unet_model.pth"))
    validate_fn(test_loader, model, loss_fn, DEVICE)

if __name__ == "__main__":
    main()
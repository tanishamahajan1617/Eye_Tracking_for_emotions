
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from tqdm import tqdm  # Batch-wise tracking ke liye


# Targetting your custom structure
from custom_datasets.Lpw import LPWGazeDataset , transformations
from Models.gaze_model import GazeModel

def main():
    #  Hardware Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"System Processing Unit: {device}")

 

    CSV_PATH = "data/lpw_extracted/labels.csv"
    IMG_DIR = "data/lpw_extracted/images"

    if not os.path.exists(CSV_PATH):
        print(f"Error: Labels sheet not found at {CSV_PATH}")
        return

    #  Dataset & Split 
    full_dataset = LPWGazeDataset(CSV_PATH, IMG_DIR, transformations)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Dataset Loaded | Train Frames: {train_size} | Val Frames: {val_size}")

    #DATALOADERS 
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )

    #Model, Loss, Optimizer Initialization
    model = GazeModel().to(device)
    criterion = nn.L1Loss()
   
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=1e-4)

    EPOCHS = 20
    best_val_loss = float("inf")

    print("\nStarting Training Loop...")
    print("=" * 50)

    for epoch in range(EPOCHS):
        # ===== TRAIN PHASE =====
        model.train()
        train_loss = 0
        
        
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] Train")
        for imgs, labels in train_bar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Forward Pass
            preds = model(imgs)
            loss = criterion(preds, labels)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

        train_loss /= len(train_loader)

        # ===== VAL =====
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                preds = model(imgs)
                loss = criterion(preds, labels)

                val_loss += loss.item()
        val_loss /= len(val_loader)

        # Epoch Metrics Log
        print(f"\n Epoch {epoch+1} Results:")
        print(f"   Train Loss: {train_loss:.4f}")
        print(f"   Val Loss:   {val_loss:.4f}")

        #Save Best Model Weights
        if val_loss < best_val_loss:
            best_val_loss = val_loss
           
            torch.save(model.state_dict(), "best_gaze_model.pth")
            print("Best model checkpoint updated & saved safely!")
        print("-" * 50)
        import gc
             
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_datasets.Lpw import LPWGazeDataset, transformations
from Models.gaze_model import GazeModel


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # DATA
    train_dataset = LPWGazeDataset("data/lpw_extracted/train.csv", transformations)
    val_dataset   = LPWGazeDataset("data/lpw_extracted/val.csv", transformations)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=32, num_workers=0)

    # MODEL
    model = GazeModel().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    EPOCHS = 10
    best_loss = float("inf")

    # TRAIN LOOP
    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # VALIDATION
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

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "Models/best_gaze_model.pth")
            print("Best model saved!")


if __name__ == "__main__":
    main()
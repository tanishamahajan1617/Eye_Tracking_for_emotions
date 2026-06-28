import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from custom_datasets.Lpw import LPWGazeDataset, transformations
from Models.gaze_model import GazeModel


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

   
    test_dataset = LPWGazeDataset("data/lpw_extracted/test.csv", transformations)

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0  
    )

    # ===== MODEL =====
    model = GazeModel().to(device)
    model.load_state_dict(torch.load("Models/best_gaze_model.pth"))
    model.eval()

    # ===== METRICS =====
    mse_total = 0
    mae_total = 0
    dist_total = 0
    total_samples = 0

    # ===== TEST LOOP =====
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            preds = model(imgs)

            
            mse = ((preds - labels) ** 2).mean(dim=1)
            mae = torch.abs(preds - labels).mean(dim=1)
            dist = torch.sqrt(((preds - labels) ** 2).sum(dim=1))

            mse_total += mse.sum().item()
            mae_total += mae.sum().item()
            dist_total += dist.sum().item()
            total_samples += labels.size(0)

    # ===== FINAL RESULTS =====
    print("\n===== TEST RESULTS =====")
    print("Test MSE:", mse_total / total_samples)
    print("Test MAE:", mae_total / total_samples)
    print("Avg Distance:", dist_total / total_samples)


if __name__ == "__main__":
    main() 
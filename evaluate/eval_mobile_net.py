import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from custom_datasets.Lpw import LPWGazeDataset, transformations
from Models.gaze_model import GazeModel


def main():

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        IMG_DIR = "data/lpw_extracted/images"
        test_dataset = LPWGazeDataset("data/test.csv", IMG_DIR, transform=transformations)
        test_loader = DataLoader(test_dataset, batch_size=16,num_workers=4, shuffle=False)


        # ===== LOAD MODEL =====
        model = GazeModel().to(device)
        model.load_state_dict(torch.load("best_gaze_model.pth"))
        model.eval()

        
        # ===== METRICS =====
        mse_total = 0
        mae_total = 0
        dist_total = 0
        count = 0


        # ===== TEST LOOP =====
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                preds = model(imgs)

                mse = torch.mean((preds - labels) ** 2)
                mae = torch.mean(torch.abs(preds - labels))
                dist = torch.sqrt(torch.sum((preds - labels) ** 2, dim=1)).mean()

                mse_total += mse.item()
                mae_total += mae.item()
                dist_total += dist.item()
                count += 1


        # ===== PRINT RESULTS =====
        print("\n===== TEST RESULTS =====")
        print("Test MSE:", mse_total / count)
        print("Test MAE:", mae_total / count)
        print("Avg Distance:", dist_total / count)



if __name__ == "__main__":
    main()   
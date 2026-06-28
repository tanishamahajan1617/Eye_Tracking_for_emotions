import torch.nn as nn
import torchvision.models as models

class GazeModel(nn.Module):
    def __init__(self):
        super().__init__()

        weights = models.MobileNet_V2_Weights.DEFAULT
        self.model = models.mobilenet_v2(weights=weights)

        in_features = self.model.classifier[1].in_features

        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()  
        )

    def forward(self, x):
        return self.model(x)
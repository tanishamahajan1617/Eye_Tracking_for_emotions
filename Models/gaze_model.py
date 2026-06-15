import torch.nn as nn
import torchvision.models as models

class GazeModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

        # grayscale fix
        self.backbone.features[0][0] = nn.Conv2d(1, 32, 3, 2, 1, bias=False)

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)  
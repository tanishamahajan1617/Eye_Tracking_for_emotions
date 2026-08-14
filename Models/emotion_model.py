import torch
import torch.nn as nn

class EmotionLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, num_classes=4, dropout=0.2):
        super(EmotionLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length=30, input_size=3)
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]  # 30th frame ka output
        logits = self.fc(last_out)
        return logits
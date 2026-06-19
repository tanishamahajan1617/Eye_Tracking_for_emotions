import torch
import torch.nn as nn

class EmotionLSTM(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_layers=2, num_classes=4):
        """
        Standalone LSTM Classifier for Sequential Emotion Tracking.
        Inputs: Shape [Batch, Seq_Len, Features] -> Standard
        """
        super(EmotionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Core LSTM cells: batch_first=True tells PyTorch that axis 0 is the Batch
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        
        # Fully Connected Classifier Head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        # Initialize Hidden (h0) and Cell states (c0) with zeros for each forward pass
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through LSTM layers -> output shape: [Batch, Seq_Len, Hidden_Size]
        out, _ = self.lstm(x, (h0, c0))
        
        # Many-to-One: Extract only the representation of the very last frame (t=29)
        # out[:, -1, :] slices the last index of the sequential dimension
        out = self.fc(out[:, -1, :])
        return out
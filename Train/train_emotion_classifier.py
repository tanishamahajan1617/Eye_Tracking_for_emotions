import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # Scaler save karne ke liye

# 1. Dataset Loading & Reshaping
df = pd.read_csv('data/synthetic/synthetic_emotion_dataset.csv')

feature_cols = ['gaze_x', 'gaze_y', 'pupil_size']

sequences = []
labels = []

for seq_id, group in df.groupby('sequence_id'):
    feat = group[feature_cols].values
    label = group['label'].iloc[0]
    sequences.append(feat)
    labels.append(label)

X = np.array(sequences, dtype=np.float32)  # Shape: (6000, 30, 3)
y = np.array(labels, dtype=np.int64)        # Shape: (6000,)

# 2. IMPORTANT FIX: Feature Standardization (Z-score Normalization)
N, T, F = X.shape
scaler = StandardScaler()
X_flat = X.reshape(-1, F)
X_scaled_flat = scaler.fit_transform(X_flat)
X = X_scaled_flat.reshape(N, T, F)

# Scaler ko save kar lein taaki live inference (app.py) me bhi use kar sakein
joblib.dump(scaler, 'gaze_scaler.pkl')

# Train-Val Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

class GazeDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(GazeDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(GazeDataset(X_val, y_val), batch_size=64, shuffle=False)

# 3. Model Architecture
class EmotionLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = EmotionLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Training Loop
epochs = 15
for epoch in range(epochs):
    model.train()
    total_loss, correct = 0, 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == targets).sum().item()
        
    val_correct = 0
    model.eval()
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            val_correct += (outputs.argmax(dim=1) == targets).sum().item()
            
    print(f"Epoch {epoch+1:02d}/{epochs} | Train Acc: {correct/len(X_train):.4f} | Val Acc: {val_correct/len(X_val):.4f}")

# Save Model Weights
torch.save(model.state_dict(), 'best_emotion_lstm.pth')
print("Model saved successfully as 'best_emotion_lstm.pth' and scaler as 'gaze_scaler.pkl'!")
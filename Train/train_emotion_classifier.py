import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os


from custom_datasets.emotion_dataset import CSVEmotionDataset
from Models.emotion_model import EmotionLSTM


BATCH_SIZE = 32
INPUT_SIZE = 6    # [gaze_x, gaze_y, pupil_size, head_yaw, head_pitch, head_roll]
HIDDEN_SIZE = 64
NUM_LAYERS = 2
NUM_CLASSES = 4   # 0: Neutral, 1: Frustrated, 2: Bored, 3: Confident
EPOCHS = 25
LR = 0.001
CSV_PATH = "data/synthetic/emotion_data.csv"

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Active Hardware Device: {device}")
    
   
    dataset = CSVEmotionDataset(CSV_PATH)
    total_len = len(dataset)
    
    
    train_size = int(0.70 * total_len)
    val_size = int(0.15 * total_len)
    test_size = total_len - train_size - val_size 
    
    # Randomly split the whole dataset into three parts
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"📊 Dataset Split Details:")
    print(f"Train Sequences: {len(train_dataset)}\n")
    print(f"Val Sequences:   {len(val_dataset)}\n")
    print(f"Test Sequences:  {len(test_dataset)}\n")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    #  2. Model Instance Invoke
    model = EmotionLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print("\nTraining LSTM Emotion Engine Started...")
    print("=" * 60)
    
    best_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        # --- TRAINING PHASE ---
        model.train()
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            total_train += batch_y.size(0)
            correct_train += predicted.eq(batch_y).sum().item()
            
        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item() * batch_X.size(0)
                _, predicted = outputs.max(1)
                total_val += batch_y.size(0)
                correct_val += predicted.eq(batch_y).sum().item()
                
        # Metrics Calculation
        epoch_train_loss = train_loss / total_train
        epoch_val_loss = val_loss / total_val
        train_acc = (correct_train / total_train) * 100
        val_acc = (correct_val / total_val) * 100
        
        print(f"Epoch [{epoch:02d}/{EPOCHS}] | Train Loss: {epoch_train_loss:.4f} (Acc: {train_acc:.2f}%) | Val Loss: {epoch_val_loss:.4f} (Acc: {val_acc:.2f}%)")
        
        # Best accuracy base checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_emotion_lstm.pth")
            
    print("=" * 60)
    print(f" Training Done! Best Validation Accuracy: {best_acc:.2f}%")
    print(" Weights saved beautifully at: 'best_emotion_lstm.pth'")
    
 
    print("\nEvaluating Model on Final Unseen Test Dataset...")
    print("-" * 60)
    
    # Load the best saved model weights for testing
    model.load_state_dict(torch.load("best_emotion_lstm.pth"))
    model.eval()
    
    test_loss, correct_test, total_test = 0.0, 0, 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            test_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            total_test += batch_y.size(0)
            correct_test += predicted.eq(batch_y).sum().item()
            
    final_test_loss = test_loss / total_test
    final_test_acc = (correct_test / total_test) * 100
    
    print(f"Test Loss: {final_test_loss:.4f} | Test Accuracy: {final_test_acc:.2f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import GestureDataset
from model import DualInputClassifier

# 訓練
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 參數設定
    BATCH_SIZE = 64
    EPOCHS = 10
    LR = 0.001 
    
    # 準備 Dataset 與 DataLoader
    csv_path = "processed_data/dataset_index.csv"
    img_dir = "processed_data/images"
    
    full_dataset = GestureDataset(csv_path, img_dir, is_train=True)
    
    # data split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # validation set 的 augmentation
    val_dataset.dataset.is_train = True 
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    model = DualInputClassifier().to(device)
    
        
    # 1. 訓練時：使用標準的 CrossEntropyLoss，確保梯度穩定
    class_weights = torch.tensor([1.0, 5.0, 5.0, 5.0, 5.0, 5.0]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    os.makedirs("model", exist_ok=True)
    
    # 2.選擇validation set中「總分」最大者當成best
    best_val_score = -float('inf')
    
    for epoch in range(EPOCHS):
        # ==========================================
        # 訓練階段 (Training Phase)
        # ==========================================
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]")
        for images, landmarks, labels in progress_bar:
            images, landmarks, labels = images.to(device), landmarks.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, landmarks)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # ==========================================
        # 驗證階段 (Validation Phase)
        # ==========================================
        model.eval()
        val_loss = 0.0
        total_score = 0.0  # 用來記錄分數
        max_possible_score = 0.0 # 記錄理論滿分
        cnt = 0
        with torch.no_grad():
            for images, landmarks, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                images, landmarks, labels = images.to(device), landmarks.to(device), labels.to(device)
                
                outputs = model(images, landmarks)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                
                for i in range(len(labels)):
                    gt = labels[i].item()
                    pred = predicted[i].item()
                    
                    # 只要這張圖是目標手勢 (1~5)，完美預測就能拿 1 分
                    if gt > 0:
                        max_possible_score += 1.0
                    
                    if gt > 0 and pred == gt:
                        cnt += 1
                        total_score += 1.0   # 預測正確的手勢: +1 分
                    elif gt > 0 and pred != gt:
                        total_score -= 2.0   # 目標手勢判錯 (包含變成 N/A): -2 分
                    elif gt == 0 and pred > 0:
                        total_score -= 2.0   # 把 N/A 誤判成手勢: -2 分
                    elif gt == 0 and pred == 0:
                        total_score += 0.0   # 成功判斷N/A: +0分
                        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1} Results: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | 競賽總分: {total_score} / {max_possible_score}")
        print(cnt)
        # 只在「總分」破紀錄時才覆蓋存檔
        if total_score > best_val_score:
            best_val_score = total_score
            torch.save(model.state_dict(), "model/best_model.pth")
            print(f"=> (新高分: {total_score} / {max_possible_score})")
            
if __name__ == '__main__':
    main()
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import GestureDataset
from model import DualInputClassifier

# 自定義Loss   目前不使用
class ScorePenaltyLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 定義 6x6 的矩陣 [真實類別][預測類別]
        penalty = torch.zeros(6, 6)
        for i in range(6):
            for j in range(6):
                if i == j:
                    penalty[i][j] = 0.0 # 預測正確
                elif i == 0 and j > 0:
                    penalty[i][j] = 2.0 # GT: NA -> Pred: 1~5 
                elif i > 0 and j != i:
                    if j == 0:
                        penalty[i][j] = 2.0 # GT: 1~5 -> Pred: NA 
                    else:
                        penalty[i][j] = 2.5 # GT: 1~5 -> Pred: wrong 1~5 
                        
        self.penalty = penalty.to(device)

    def forward(self, logits, targets):
        # 1. 基本的交叉熵 (確保基礎特徵學習穩定)
        base_loss = self.ce_loss(logits, targets)
        
        # 2. 計算模型對每個類別的預測機率 (Softmax)
        probs = torch.softmax(logits, dim=1)
        
        # 3. 根據 Targets 取出對應的懲罰權重
        # batch_penalties shape: [batch_size, 6]
        batch_penalties = self.penalty[targets] 
        
        # 4. 期望懲罰 = 機率 * 懲罰權重 的總和
        expected_penalty = torch.sum(probs * batch_penalties, dim=1).mean()
        
        # 結合兩者 (可調整 0.5 這個超參數，目前先設 1.0 加強懲罰)
        return base_loss + 1.0 * expected_penalty

# 訓練
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 參數設定
    BATCH_SIZE = 64
    EPOCHS = 5
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
    criterion = nn.CrossEntropyLoss(class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    
    os.makedirs("model_checkpoints", exist_ok=True)
    
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
        
        print(f"\nEpoch {epoch+1} Results: Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | ★ 競賽總分: {total_score} / {max_possible_score}")
        print(cnt)
        # 只在「總分」破紀錄時才覆蓋存檔
        if total_score > best_val_score:
            best_val_score = total_score
            torch.save(model.state_dict(), "model_checkpoints/best_model.pth")
            print(f"=> 恭喜！發現更高分的模型 (新高分: {total_score} / {max_possible_score})，已存檔！")
            
if __name__ == '__main__':
    main()
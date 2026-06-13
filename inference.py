import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
from pathlib import Path
# ---------------------------------------------------------
# 模型架構
# ---------------------------------------------------------
class DualInputClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(DualInputClassifier, self).__init__()
        
        # --- 影像分支 (CNN + CBAM Attention) ---
        # 我們在第 2 層和第 4 層卷積後方插入 CBAM，攔截並強化特徵
        self.image_extractor = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # Block 2 + Attention
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            CBAMBlock(32), 

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 4 + Attention
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAMBlock(64), 

            # 全局池化壓縮
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        img_feature_dim = 64
        
        # --- 座標分支 (Transformer) --- 
        self.landmark_extractor = HandTransformerExtractor(
            in_features=2, 
            d_model=64, 
            nhead=4, 
            num_layers=2
        )
        lm_feature_dim = 64
        
        # --- 融合與分類器 (保持不變) ---
        self.classifier = nn.Sequential(
            nn.Linear(img_feature_dim + lm_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, landmarks):
        # 1. 影像萃取
        img_feat = self.image_extractor(img)
        
        # 2. 座標萃取
        batch_size = landmarks.size(0)
        landmarks_graph = landmarks.view(batch_size, 21, 2)
        # 通過 Transformer
        lm_feat = self.landmark_extractor(landmarks_graph) 
        
        # 3. 融合輸出
        combined_feat = torch.cat((img_feat, lm_feat), dim=1)
        out = self.classifier(combined_feat)
        return out
    
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 利用 1x1 Conv 取代 Linear 來做維度壓縮與還原，節省參數
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # 將 Avg 和 Max Pooling 結果串接後，用一層卷積生出空間權重圖
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAMBlock(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        # 先乘上通道權重，再乘上空間權重
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x
    
class HandTransformerExtractor(nn.Module):
    """手部專用 Transformer 特徵萃取器"""
    def __init__(self, in_features=2, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        """
        參數:
        in_features: 輸入維度 (x, y 座標 = 2)
        d_model: Transformer 內部運作的特徵維度 (對應原 GCN 的 out_features=64)
        nhead: 多頭注意力的頭數
        num_layers: Transformer Encoder 的層數 (對應原 GCN 有 2 層)
        dim_feedforward: FFN 的隱藏層維度
        """
        super(HandTransformerExtractor, self).__init__()
        
        # 1. 座標映射 (Linear Projection)
        # 將低維度的 2D 座標投射到高維度的 d_model 空間，讓 Transformer 更好處理
        self.embedding = nn.Linear(in_features, d_model)
        
        # 2. 位置編碼 (Learnable Positional Embedding)
        # MediaPipe 固定有 21 個節點，這裡建立一個可學習的參數矩陣 (1, 21, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 21, d_model))
        
        # 3. Transformer Encoder
        # batch_first=True 可以讓輸入和輸出的 shape 維持 (Batch, Seq_len, Feature)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=0.2,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x 原始 shape: (Batch, 21, 2)
        
        # 投射到高維特徵空間
        x = self.embedding(x)  # shape: (Batch, 21, d_model)
        
        # 加入位置編碼 (利用 broadcasting 自動擴展到符合 Batch size)
        x = x + self.pos_embedding  # shape: (Batch, 21, d_model)
        
        # 進入 Transformer 進行 Self-Attention 計算
        x = self.transformer(x)  # shape: (Batch, 21, d_model)
        
        # 全局平均池化 (Global Average Pooling)
        # 將 21 個節點的特徵壓縮成一個 d_model 維度的向量，準備與影像特徵融合
        x = torch.mean(x, dim=1)  # shape: (Batch, d_model)
        
        return x
# ---------------------------------------------------------
# 2. 全域初始化：載入模型與設定
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model = DualInputClassifier().to(device)

try:
    base_dir = Path(__file__).parent
except NameError:
    base_dir = Path.cwd()
weights_path = base_dir / 'model' / 'best_model.pth'
model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()

# 影像預處理 (必須跟訓練時的 transform 一致)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---------------------------------------------------------
# 3. Predict 
# ---------------------------------------------------------
def predict(cropped_img: np.ndarray, landmarks: np.ndarray) -> int:
    """
    Args:
        cropped_img: RGB image array (H, W, 3)
        landmarks: numpy array shape (21, 2)
    Returns:
        final_decision_class: int {0,1,2,3,4,5}
    """
    try:
            
        # 影像轉換
        img_resized = cv2.resize(cropped_img, (128, 128), interpolation=cv2.INTER_AREA)
        img_pil = Image.fromarray(img_resized)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        # landmark座標平移
        norm_landmarks = landmarks.copy().astype(np.float32)
        wrist = norm_landmarks[0].copy()
        
        # 1. 平移：將手腕變成 (0, 0)
        norm_landmarks = norm_landmarks - wrist
        
        # 2. 縮放：壓縮到 -1.0 ~ 1.0 之間
        max_val = np.max(np.abs(norm_landmarks))
        if max_val > 0:
            norm_landmarks = norm_landmarks / max_val
        
        # 座標轉換：使用正規化後的 norm_landmarks 攤平成 42 維
        lm_tensor = torch.tensor(norm_landmarks.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
        
        # 推論
        with torch.no_grad():
            outputs = model(img_tensor, lm_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
        predicted_class = int(np.argmax(probs))
        # confidence = probs[predicted_class]
                
        return predicted_class
        
    except Exception as e:
        return 0
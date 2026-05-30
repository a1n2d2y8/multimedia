import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ---------------------------------------------------------
# 模型架構
# ---------------------------------------------------------
class DualInputClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(DualInputClassifier, self).__init__()
        
        # 影像分支 (Image Extractor)
        # 使用預訓練的 MobileNetV3-Small，拔除最後的分類層，提取 576 維特徵
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.image_extractor = nn.Sequential(
            mobilenet.features,
            mobilenet.avgpool,
            nn.Flatten()
        )
        # MobileNetV3-Small 的輸出特徵維度是 576
        img_feature_dim = 576
        
        # 座標分支 (Landmark Extractor)
        # 輸入為 42 維 (21個點 x 2)，輸出 128 維特徵
        self.landmark_extractor = nn.Sequential(
            nn.Linear(42, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        lm_feature_dim = 128
        
        # 融合與分類器 (Fusion & Classifier)
        # 576 + 128 = 704 維
        self.classifier = nn.Sequential(
            nn.Linear(img_feature_dim + lm_feature_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3), # 防止過擬合
            nn.Linear(256, num_classes)
        )

    def forward(self, img, landmarks):
        # 分別萃取特徵
        img_feat = self.image_extractor(img)
        lm_feat = self.landmark_extractor(landmarks)
        
        # 將兩種特徵串接 (Concatenate)
        combined_feat = torch.cat((img_feat, lm_feat), dim=1)
        
        # 輸出 6 個類別 (0~5) 的 Logits
        out = self.classifier(combined_feat)
        return out
# ---------------------------------------------------------
# 2. 全域初始化：載入模型與設定
# ---------------------------------------------------------
device = torch.device("cuda") 
model = DualInputClassifier().to(device)

# 讀取相對路徑下的權重 
weights_path = os.path.join(os.path.dirname(__file__), 'model_checkpoints', 'best_model.pth')
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
        import cv2
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

        return predicted_class
        
    except Exception as e:
        return 0
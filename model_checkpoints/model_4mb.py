import torch
import torch.nn as nn
import torchvision.models as models

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
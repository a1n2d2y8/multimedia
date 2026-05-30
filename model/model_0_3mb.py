import torch
import torch.nn as nn

class DualInputClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(DualInputClassifier, self).__init__()
        
        # 影像分支 (Image Extractor)
        # 改用極度輕量化的 4 層 CNN
        # 總參數量不到 6 萬，檔案大小極小，但對於 128x128 的手部特徵提取非常足夠
        self.image_extractor = nn.Sequential(
            # Input: 3 x 128 x 128 -> Output: 16 x 64 x 64
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # 16 x 64 x 64 -> Output: 32 x 32 x 32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 32 x 32 x 32 -> Output: 64 x 16 x 16
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 64 x 16 x 16 -> Output: 64 x 8 x 8
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 全局池化，將 64x8x8 壓縮成 64 維向量
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        img_feature_dim = 64
        
        # 座標分支 (Landmark Extractor)
        # 維持 MLP 結構，但將隱藏層維度微調至 64
        self.landmark_extractor = nn.Sequential(
            nn.Linear(42, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True)
        )
        lm_feature_dim = 64
        
        # 融合與分類器 (Fusion & Classifier)
        # 64 + 64 = 128 維
        self.classifier = nn.Sequential(
            nn.Linear(img_feature_dim + lm_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
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
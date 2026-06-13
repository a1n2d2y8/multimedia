import torch
import torch.nn as nn

class DualInputClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(DualInputClassifier, self).__init__()
        
        # --- 影像分支 (Image Extractor) ---
        # 保持你的極度輕量化 4 層 CNN (這個設計非常棒，完全符合規格書的輕量化要求)
        self.image_extractor = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        img_feature_dim = 64
        
        # --- 座標分支 (Landmark Extractor) ---
        # ★ 替換為我們剛剛寫的 HandGCNExtractor
        self.landmark_extractor = HandGCNExtractor(in_features=2, hidden_dim=32, out_features=64)
        lm_feature_dim = 64
        
        # --- 融合與分類器 (Fusion & Classifier) ---
        self.classifier = nn.Sequential(
            nn.Linear(img_feature_dim + lm_feature_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, landmarks):
        # 1. 萃取影像特徵 -> (Batch, 64)
        img_feat = self.image_extractor(img)
        
        # ★ 2. 處理座標特徵
        # 將輸入的 (Batch, 42) 重塑為圖結構需要的 (Batch, 21個節點, 2個座標)
        batch_size = landmarks.size(0)
        landmarks_graph = landmarks.view(batch_size, 21, 2)
        
        # 萃取 GCN 特徵 -> (Batch, 64)
        lm_feat = self.landmark_extractor(landmarks_graph)
        
        # 3. 融合與輸出
        combined_feat = torch.cat((img_feat, lm_feat), dim=1)
        out = self.classifier(combined_feat)
        
        return out
    
class GraphConvolution(nn.Module):
    """基礎的圖卷積層"""
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        # x: (Batch, 21, in_features)
        # adj: (21, 21)
        support = torch.matmul(x, self.weight)
        # 沿著骨架連接關係傳遞特徵
        output = torch.matmul(adj, support)
        return output + self.bias

class HandGCNExtractor(nn.Module):
    """手部專用 GCN 特徵萃取器"""
    def __init__(self, in_features=2, hidden_dim=32, out_features=64):
        super(HandGCNExtractor, self).__init__()
        
        self.gcn1 = GraphConvolution(in_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)
        
        # 註冊鄰接矩陣為 buffer，這樣它會隨著模型自動存檔與移動 (CPU/GPU)
        self.register_buffer('adj', self._build_adjacency_matrix())

    def _build_adjacency_matrix(self):
        # MediaPipe 手部 21 節點的物理連接 (Edges)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4),        # 大拇指
            (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
            (0, 9), (9, 10), (10, 11), (11, 12),   # 中指
            (0, 13), (13, 14), (14, 15), (15, 16), # 無名指
            (0, 17), (17, 18), (18, 19), (19, 20), # 小拇指
            (5, 9), (9, 13), (13, 17)              # 手掌內部橫向連接 (強化掌心結構)
        ]
        
        A = torch.zeros(21, 21)
        for i, j in edges:
            A[i, j] = 1.0
            A[j, i] = 1.0
        A += torch.eye(21) # 加上 Self-loops (讓節點也保留自己的特徵)

        # 對稱正規化 (Symmetric Normalization): D^{-1/2} * A * D^{-1/2}
        D = torch.sum(A, dim=1)
        D_inv_sqrt = torch.pow(D, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
        D_mat_inv_sqrt = torch.diag(D_inv_sqrt)
        A_norm = torch.matmul(torch.matmul(D_mat_inv_sqrt, A), D_mat_inv_sqrt)
        
        return A_norm

    def forward(self, x):
        # x 原始 shape: (Batch, 21, 2)
        
        # Layer 1
        x = self.gcn1(x, self.adj)
        x = x.transpose(1, 2) # BatchNorm1d 需要 shape 為 (Batch, Features, Nodes)
        x = self.bn1(x)
        x = x.transpose(1, 2)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.gcn2(x, self.adj)
        x = x.transpose(1, 2)
        x = self.bn2(x)
        x = x.transpose(1, 2)
        x = self.relu(x)
        
        # 全局平均池化 (Global Average Pooling)：把 21 個節點的特徵壓縮成一個 64 維的向量
        x = torch.mean(x, dim=1) 
        return x

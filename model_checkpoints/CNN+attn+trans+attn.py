import torch
import torch.nn as nn

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
            CBAMBlock(32),  # 👈 第一道注意力防線：篩選初階邊緣/紋理

            # Block 3
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 4 + Attention
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            CBAMBlock(64),  # 👈 第二道注意力防線：聚焦高階手勢語意

            # 全局池化壓縮
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        img_feature_dim = 64
        
        # --- 座標分支 (換成 Transformer) --- 
        # 👇 替換成新的 Transformer 萃取器
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
        # 1. 影像特徵 (Batch, 64)
        img_feat = self.image_extractor(img)
        
        # 2. 座標特徵，注意這裡必須輸出 (Batch, 21, 64)，不要做 Pooling
        batch_size = landmarks.size(0)
        landmarks_graph = landmarks.view(batch_size, 21, 2)
        lm_feat_seq = self.landmark_extractor(landmarks_graph) 
        
        # 3. 交叉注意力融合 (Image 當 Q, Landmark 當 K,V)
        # 輸出會是 (Batch, 64) 的高階融合特徵
        fused_feat = self.fusion_module(img_feat, lm_feat_seq)
        
        # 4. 最終分類 (這裡分類器的 input_dim 要改成 64，而不是原本的 128)
        out = self.classifier(fused_feat)
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

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.2):
        super(CrossAttentionFusion, self).__init__()
        
        # 核心：多頭交叉注意力機制
        # batch_first=True 讓輸入形狀為 (Batch, Seq, Feature)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization 幫助收斂
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # FFN (前饋神經網路)，進一步轉換融合後的特徵
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, img_feat, lm_feat_seq):
        # 輸入形狀確認：
        # img_feat: (Batch, 64) -> 這是整張影像的全局特徵
        # lm_feat_seq: (Batch, 21, 64) -> 這是 21 個節點各自的特徵
        
        # 1. 準備 Q, K, V
        # 將 Image 增加一個序列維度，變成 (Batch, 1, 64) 作為 Query
        Q = img_feat.unsqueeze(1) 
        K = lm_feat_seq
        V = lm_feat_seq
        
        # 2. 進行 Cross-Attention
        # attn_output 形狀會是 (Batch, 1, 64)
        attn_output, attn_weights = self.cross_attn(Q, K, V)
        
        # 3. 殘差連接與正規化 (Add & Norm)
        # 把原始的影像特徵加回去，確保基本盤資訊不流失
        x = self.norm1(Q + attn_output)
        
        # 4. FFN 與第二次殘差
        ffn_output = self.ffn(x)
        out = self.norm2(x + ffn_output)
        
        # 5. 壓平回 1D 向量 (Batch, 64) 交給最後的分類器
        return out.squeeze(1)
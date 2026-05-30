import os
import cv2
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

# =====================================================================
# 1. 把你目前正在使用的 DualInputClassifier 貼在這裡
# (以下示範為你上一篇的 MobileNetV3 + GRU 架構，請確保與你訓練時完全一致)
# =====================================================================
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
        img_feature_dim = 576
        
        # 座標分支 (Landmark Extractor) - 改用 GRU
        # 輸入為 21 個時間步 (Sequence Length=21)，每個時間步特徵為 2 (x, y)
        self.gru_hidden_size = 64
        self.landmark_extractor = nn.GRU(
            input_size=2,          # 每個點有 (x, y) 兩個座標
            hidden_size=self.gru_hidden_size,
            num_layers=1,          # GRU 層數
            batch_first=True,      # 讓輸入的第一個維度是 batch_size
            bidirectional=True     # 開啟雙向 GRU (效果通常比單向好)
        )
        # 因為是雙向，輸出的特徵維度會是 hidden_size * 2 = 128，剛好銜接你原本的架構
        lm_feature_dim = self.gru_hidden_size * 2
        
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
        # 1. 萃取影像特徵
        # img_feat shape: (batch_size, 576)
        img_feat = self.image_extractor(img)
        
        # 2. 處理座標特徵並餵給 GRU
        batch_size = landmarks.size(0)
        # 將 (batch_size, 42) 重塑為 (batch_size, 21, 2)
        landmarks_seq = landmarks.view(batch_size, 21, 2)
        
        # GRU 會回傳 output 和最後的隱藏狀態 hidden
        # hidden 的形狀為 (num_layers * num_directions, batch_size, hidden_size)
        _, hidden = self.landmark_extractor(landmarks_seq)
        
        # 提取雙向 GRU 最後一層的正向與反向隱藏狀態，並把它們串接起來
        # hidden[-2, :, :] 是正向最後狀態，hidden[-1, :, :] 是反向最後狀態
        lm_feat = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        
        # 3. 將兩種特徵串接 (Concatenate)
        # combined_feat shape: (batch_size, 576 + 128)
        combined_feat = torch.cat((img_feat, lm_feat), dim=1)
        
        # 4. 輸出預測
        out = self.classifier(combined_feat)
        return out

# =====================================================================
# 2. 參數與路徑設定
# =====================================================================
CSV_PATH = Path("processed_data/dataset_index.csv")
IMG_BASE_DIR = Path("processed_data/images")
MODEL_WEIGHTS = "model_checkpoints/best_model.pth"

# 輸出結果的資料夾
OUTPUT_DIR = Path("sample_test_results")
OUT_IMG_DIR = OUTPUT_DIR / "images"
OUT_CSV_PATH = OUTPUT_DIR / "predictions.csv"

# 我們要抽樣的類別
TARGET_CLASSES = ['fist', 'one', 'like', 'palm', 'ok']
SAMPLES_PER_CLASS = 10

# 類別 ID 反查表
ID_TO_CLASS = {
    0: "N/A",
    1: "fist",
    2: "like",
    3: "ok",
    4: "one",
    5: "palm"
}

def main():
    # 建立輸出資料夾
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")
    
    # 載入模型
    model = DualInputClassifier().to(device)
    if not os.path.exists(MODEL_WEIGHTS):
        print(f"❌ 找不到模型權重檔: {MODEL_WEIGHTS}")
        return
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()
    
    # 影像轉換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("讀取 CSV 並進行隨機抽樣...")
    df = pd.read_csv(CSV_PATH)
    
    # 從每個目標類別中隨機抽取 10 筆
    sampled_dfs = []
    for cls in TARGET_CLASSES:
        df_cls = df[df['original_class'] == cls]
        if len(df_cls) == 0:
            print(f"⚠️ 警告：找不到類別 {cls} 的資料！")
            continue
            
        n_samples = min(SAMPLES_PER_CLASS, len(df_cls))
        sampled_dfs.append(df_cls.sample(n=n_samples, random_state=42))
        
    df_sample = pd.concat(sampled_dfs).reset_index(drop=True)
    print(f"共抽出 {len(df_sample)} 筆資料，開始進行預測！")
    
    results = []
    
    for idx, row in df_sample.iterrows():
        orig_class = row['original_class']
        img_rel_path = row['filename'] # 例如: fist/fist_123.jpg
        full_img_path = IMG_BASE_DIR / img_rel_path
        
        if not full_img_path.exists():
            print(f"⚠️ 找不到圖片: {full_img_path}，跳過。")
            continue
            
        # 1. 處理影像
        img_bgr = cv2.imread(str(full_img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = transform(img_pil).unsqueeze(0).to(device)
        
        # 2. 處理座標特徵
        lm_cols = [f'lm_{i}' for i in range(42)]
        landmarks = row[lm_cols].values.astype(np.float32)
        
        # 💡 如果你在 dataset.py 中有做「手腕相對座標正規化」，請解除下方註解
        wrist_x, wrist_y = landmarks[0], landmarks[1]
        landmarks[0::2] -= wrist_x
        landmarks[1::2] -= wrist_y
        max_val = np.max(np.abs(landmarks))
        if max_val > 0:
            landmarks = landmarks / max_val
            
        lm_tensor = torch.tensor(landmarks).unsqueeze(0).to(device)
        
        # 3. 模型推論
        with torch.no_grad():
            outputs = model(img_tensor, lm_tensor)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
        pred_class_id = int(np.argmax(probs))
        confidence = float(probs[pred_class_id])
        pred_class_name = ID_TO_CLASS.get(pred_class_id, "Unknown")
        
        is_correct = (orig_class == pred_class_name)
        
        # 4. 記錄預測結果
        results.append({
            'Filename': full_img_path.name,
            'True Class': orig_class,
            'Pred Class': pred_class_name,
            'Confidence': round(confidence, 4),
            'Is Correct': is_correct
        })
        
        # 5. 複製圖片並標示結果 (為了方便查看，我們把預測結果寫在檔名上)
        # 檔名格式: [True]_fist_[Pred]_fist_0.95_fist_123.jpg
        status_icon = "O" if is_correct else "X"
        
        # ★ 新增：將類別名稱中的斜線替換成底線，避免破壞檔名路徑
        safe_orig_class = orig_class.replace("/", "_")
        safe_pred_class = pred_class_name.replace("/", "_")
        
        # 使用安全的名稱來組合檔名
        save_filename = f"[{status_icon}]_True_{safe_orig_class}_Pred_{safe_pred_class}_{confidence:.2f}_{full_img_path.name}"
        save_filepath = OUT_IMG_DIR / save_filename
        
        # 將原始裁切圖複製過去
        shutil.copy(str(full_img_path), str(save_filepath))
        
    # 儲存 CSV 報表
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT_CSV_PATH, index=False, encoding='utf-8-sig')
    
    # 計算並印出準確率
    acc = result_df['Is Correct'].mean() * 100
    print("\n" + "="*50)
    print(f"✅ 抽樣預測完成！50 張圖片的準確率: {acc:.2f}%")
    print(f"📊 詳細報表已儲存至: {OUT_CSV_PATH}")
    print(f"🖼️ 抽樣圖片已儲存至: {OUT_IMG_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()
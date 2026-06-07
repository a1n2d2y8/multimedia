import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class GestureDataset(Dataset):
    def __init__(self, csv_path, img_dir, is_train=True):
        # 1. 讀取完整的資料
        full_data = pd.read_csv(csv_path)
        
        if is_train:
            # 挑出 目標類別 (1~5) 和 N/A 類別 (0)
            df_targets = full_data[full_data['label'] != 0]
            df_na = full_data[full_data['label'] == 0]
            
            # 計算目標類別的總數
            target_count = len(df_targets)
            
            # 對 N/A 進行欠採樣：
            # 設定 N/A 的數量等於「目標類別總和」的X倍
            sample_size = int(target_count* 2.0) 
            
            # 如果 N/A 本來就比 sample_size 少，就全拿；否則隨機抽樣
            if len(df_na) > sample_size:
                df_na_sampled = df_na.sample(n=sample_size, random_state=42)
            else:
                df_na_sampled = df_na
                
            # 將兩組資料合併，並打亂
            self.data = pd.concat([df_targets, df_na_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
            
            print(f"   -> 1~5 類別數量: {len(df_targets)}")
            print(f"   -> N/A 類別數量 (抽樣後): {len(df_na_sampled)}")
            print(f"   -> 總資量 {len(full_data)} -> {len(self.data)}")
            
        else:
            # 驗證集不需要平衡，直接使用
            self.data = full_data

        self.img_dir = img_dir
        self.is_train = is_train
        
        # data augmentation
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 1. 讀取影像
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        
        # 2. 讀取 42 維 Landmark (確保轉成 float32)
        lm_cols = [f'lm_{i}' for i in range(42)]
        landmarks = row[lm_cols].values.astype(np.float32)
        
        # --- 正規化 ---
        # MediaPipe 的第 0 個點 (x0, y0) 是手腕 (Wrist)
        wrist_x = landmarks[0]
        wrist_y = landmarks[1]
        
        # 1. 將所有點平移，讓手腕變成 (0, 0)
        landmarks[0::2] -= wrist_x  # 所有 X 座標減去手腕 X
        landmarks[1::2] -= wrist_y  # 所有 Y 座標減去手腕 Y
        
        # 2. 找出最大的絕對數值來做 Scaling
        # 確保所有座標都被完美壓縮在 -1.0 到 1.0 之間，且不破壞幾何形狀
        max_val = np.max(np.abs(landmarks))
        if max_val > 0:
            landmarks = landmarks / max_val
        
        landmarks = torch.tensor(landmarks)
        label = int(row['label'])
        
        return image, landmarks, label
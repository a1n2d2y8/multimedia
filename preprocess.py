import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
# 匯入助教提供的預處理模組
from hand_preprocess import MediaPipeHandPreprocessor

# --- 參數設定 ---
DATASET_DIR = Path("HaGRIDv2_dataset_512")
OUTPUT_DIR = Path("processed_data")
IMG_OUTPUT_DIR = OUTPUT_DIR / "images"
TARGET_IMG_SIZE = (128, 128) # 神經網路需要的固定輸入大小
CSV_PATH = OUTPUT_DIR / "dataset_index.csv"

# 定義 Target Classes 對應的 Label (1~5)，其餘皆為 0
TARGET_CLASSES = {
    'fist': 1,
    'like': 2,
    'ok': 3,
    'one': 4,
    'palm': 5
}

def main():
    # 建立輸出總資料夾
    IMG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 確保一開始先刪除舊的 CSV，避免重複執行程式時，資料一直疊加在檔案後面
    if CSV_PATH.exists():
        CSV_PATH.unlink()
        print(f"已清除舊的索引檔: {CSV_PATH}")
    
    total_processed = 0
    total_skipped = 0 # ★ 用來統計因為多隻手而被過濾掉的圖片數量
    
    # 設定 max_num_hands=2，讓模型偵測是否有第二隻手
    with MediaPipeHandPreprocessor(max_num_hands=2) as preprocessor:
        
        if not DATASET_DIR.exists():
            print(f"找不到資料夾: {DATASET_DIR}")
            return
            
        class_folders = [f for f in DATASET_DIR.iterdir() if f.is_dir()]
        
        for folder in class_folders:
            class_name = folder.name
            label = TARGET_CLASSES.get(class_name, 0)
            
            print(f"\n--- 開始處理類別: {class_name} -> Label: {label} ---")
            
            # ★ 新增：為目前類別建立專屬的子資料夾 (例如 processed_data/images/fist)
            class_out_dir = IMG_OUTPUT_DIR / class_name
            class_out_dir.mkdir(parents=True, exist_ok=True)
            
            # 每個資料夾獨立一個暫存 list，避免記憶體爆滿
            folder_metadata = []
            
            # 取得圖片路徑
            image_paths = list(folder.glob("*.jpg")) 
            
            for img_path in tqdm(image_paths, desc=class_name):
                try:
                    # 手動使用 PIL 讀取圖片並呼叫 detect_hand
                    with Image.open(img_path) as img:
                        img = img.convert('RGB') # 確保為 RGB 格式
                        result = preprocessor.detect_hand(img)
                    
                    if result is None:
                        continue
                        
                    # 根據預處理器的回傳值解包，取得手部數量
                    crop, landmarks, _bbox, _image_landmarks, num_hands = result
                    
                    # 終極過濾器！超過一隻手就直接丟棄
                    if num_hands > 1:
                        total_skipped += 1
                        continue
                        
                    crop_resized = cv2.resize(crop, TARGET_IMG_SIZE, interpolation=cv2.INTER_AREA)
                    
                    save_filename = f"{class_name}_{img_path.stem}.jpg"
                    # ★ 更改存檔路徑：存入對應的類別資料夾
                    save_filepath = class_out_dir / save_filename
                    
                    cv2.imwrite(str(save_filepath), cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))
                    
                    flat_landmarks = landmarks.flatten().tolist()
                    
                    # ★ 在 CSV 中記錄相對路徑 (例如 fist/fist_xxx.jpg)
                    # 這樣原本的 Dataset 讀取程式就能直接 os.path.join 找到檔案
                    relative_path = f"{class_name}/{save_filename}"
                    
                    row = {
                        'filename': relative_path,
                        'label': label,
                        'original_class': class_name
                    }
                    for i, val in enumerate(flat_landmarks):
                        row[f'lm_{i}'] = val
                        
                    folder_metadata.append(row)
                    
                except Exception as e:
                    print(f"\n處理 {img_path} 時發生錯誤: {e}")
                    continue

            # --- 資料夾處理完畢：立刻寫入 CSV 並釋放記憶體 ---
            if folder_metadata:
                df = pd.DataFrame(folder_metadata)
                
                # 如果 CSV 檔案還不存在 (代表這是第一批寫入)，就寫入標題列 (header)
                # 如果已經存在，就使用 mode='a' (append) 附加在結尾，並忽略 header
                write_header = not CSV_PATH.exists()
                df.to_csv(CSV_PATH, mode='a', index=False, header=write_header)
                
                folder_count = len(df)
                total_processed += folder_count
                print(f"-> [{class_name}] 已成功寫入 {folder_count} 筆純淨資料至 CSV。暫存已清空。")

    print(f"\n全部資料預處理完成！共成功處理 {total_processed} 張圖片。")
    print(f"★ 成功過濾掉 {total_skipped} 張含有多隻手部干擾的圖片。")
    print(f"最終索引檔儲存於: {CSV_PATH}")

if __name__ == "__main__":
    main()
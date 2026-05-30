import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd 
from PIL import Image
from pathlib import Path
import torchvision.models as models
import torchvision.transforms as transforms
from hand_preprocess import MediaPipeHandPreprocessor
from inference import predict
# --- 參數設定 ---
TEST_IMG_DIR = Path("testcase") 
DEBUG_OUT_DIR = Path("debug") 
CSV_OUTPUT_PATH = "test_results.csv"

# 定義 Target Classes 對應的 Label
TARGET_CLASSES_MAP = {
    'fist': 1,
    'like': 2,
    'ok': 3,
    'one': 4,
    'palm': 5
}
def main():
    if not TEST_IMG_DIR.exists():
        print(f"請先建立 {TEST_IMG_DIR} 資料夾，並放入測試圖片！")
        return

    DEBUG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = [p for p in TEST_IMG_DIR.glob("*.*") if p.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if not image_paths:
        print(f"{TEST_IMG_DIR} 裡面沒有找到圖片喔！")
        return

    results_data = []

    with MediaPipeHandPreprocessor() as preprocessor:
        for img_path in image_paths:
            
            true_class_str = img_path.stem.split('_')[0].lower()
            target_label = TARGET_CLASSES_MAP.get(true_class_str, 0)
            
            # 1. 預處理
            result = preprocessor.preprocess_path(img_path)
            
            if result is None:
                results_data.append({
                    'Filename': img_path.name,
                    'Target Class': target_label,
                    'Predicted Class': 0, # Hand Not Detected，預測為 0
                    'Is Correct': (0 == target_label)
                })
                continue
                
            crop, landmarks = result
            
            # 儲存 debug 圖片，可以用來確認有沒有正確抓到圖片中的手
            crop_resized = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(DEBUG_OUT_DIR / f"crop_{img_path.name}"), cv2.cvtColor(crop_resized, cv2.COLOR_RGB2BGR))
            
            # 使用inference.py中的predict
            final_pred_class = predict(crop, landmarks)
            
            # 記錄結果
            is_correct = (final_pred_class == target_label)
            results_data.append({
                'Filename': img_path.name,
                'Target Class': target_label,
                'Predicted Class': final_pred_class,
                'Is Correct': is_correct
            })

    # 輸出報表
    df = pd.DataFrame(results_data)
    df.to_csv(CSV_OUTPUT_PATH, index=False, encoding='utf-8-sig') 
    
    total_imgs = len(df)
    correct_preds = df['Is Correct'].sum()
    acc = (correct_preds / total_imgs) * 100 if total_imgs > 0 else 0
    
    print("\n" + "="*50)
    print(f"評測完成！總共測試 {total_imgs} 張圖片，準確率: {acc:.2f}%")
    print(f"詳細結果已儲存至: {CSV_OUTPUT_PATH}")
    print("="*50)

if __name__ == "__main__":
    main()
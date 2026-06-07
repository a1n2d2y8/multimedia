import os
import numpy as np


_TRIPLETS = (
    (1, 0, 2), (2, 1, 3), (3, 2, 4),
    (5, 0, 6), (6, 5, 7), (7, 6, 8),
    (9, 0, 10), (10, 9, 11), (11, 10, 12),
    (13, 0, 14), (14, 13, 15), (15, 14, 16),
    (17, 0, 18), (18, 17, 19), (19, 18, 20),
)

def extract_features(landmarks):
    pts = np.asarray(landmarks, dtype=np.float64).reshape(21, 2)
    feats = np.empty(55, dtype=np.float32)
    feats[:40] = pts[1:].reshape(-1)  
    for k, (i, a, b) in enumerate(_TRIPLETS):
        v1 = pts[a] - pts[i]
        v2 = pts[b] - pts[i]
        m1, m2 = np.hypot(*v1), np.hypot(*v2)
        if m1 < 1e-9 or m2 < 1e-9:
            feats[40 + k] = 180.0
        else:
            cos = np.clip(np.dot(v1, v2) / (m1 * m2), -1.0, 1.0)
            feats[40 + k] = np.degrees(np.arccos(cos))
    return feats

class GestureClassifier:
    def __init__(self, npz_path):
        d = np.load(npz_path, allow_pickle=True)
        self._mean = d["scaler_mean"]
        self._scale = d["scaler_scale"]
        self._coefs = [c.astype(np.float32) for c in d["coefs"]]
        self._intercepts = [b.astype(np.float32) for b in d["intercepts"]]
        self._classes = d["classes"]
        self.class_names = list(d["class_names"])

    def _forward(self, x):
        a = (x - self._mean) / self._scale
        n = len(self._coefs)
        for i in range(n):
            a = a @ self._coefs[i] + self._intercepts[i]
            if i < n - 1:
                a = np.maximum(a, 0.0)  # ReLU
        return a

    def _softmax(self, z):
        e = np.exp(z - z.max())
        return e / e.sum()

    def predict_full(self, landmarks):
        logits = self._forward(extract_features(landmarks))
        j = int(np.argmax(logits))
        return (int(self._classes[j]), self.class_names[int(self._classes[j])],
                float(self._softmax(logits)[j]))

# ==========================================
# 2. 全域初始化 (載入權重)
# ==========================================
# 安全路徑寫法：永遠相對於 inference.py
try:
    base_dir = os.path.dirname(__file__)
except NameError:
    base_dir = os.getcwd()

# 假設你把 best_weight.npz 放在 model 資料夾底下
weights_path = os.path.join(base_dir, 'model', 'best_weight.npz')
clf = GestureClassifier(weights_path)

# 建立字串到整數標籤的映射表 (助教要求 0~5)
LABEL_MAP = {
    'other': 0, 
    'fist': 1, 
    'like': 2, 
    'ok': 3, 
    'one': 4, 
    'palm': 5
}

# ==========================================
# 3. 助教要求的官方介面
# ==========================================
def predict(cropped_img: np.ndarray, landmarks: np.ndarray) -> int:
    """
    Args:
        cropped_img: RGB image array (H, W, 3)
        landmarks: numpy array shape (21, 2)
    Returns:
        final_decision_class: int {0,1,2,3,4,5}
    """
    try:
        if landmarks is None or np.isnan(landmarks).any() or landmarks.shape != (21, 2):
            return 0
            
        # ★ 關鍵：將座標 Normalize 到 0..1 (符合你模型的訓練前提)
        # 根據傳進來的圖片長寬來正規化座標
        h, w = cropped_img.shape[:2]
        norm_landmarks = landmarks.copy().astype(np.float64)
        norm_landmarks[:, 0] /= w
        norm_landmarks[:, 1] /= h
        
        # 進行預測
        _, class_name, confidence = clf.predict_full(norm_landmarks)
        
        # 轉換為助教需要的整數 ID
        pred_id = LABEL_MAP.get(class_name, 0)
        
        # 🛡️ 信心度防禦 (可選)：如果你覺得模型有時候猜 1~5 不太確定
        # if pred_id != 0 and confidence < 0.65:
        #     return 0 # 強制降轉為 N/A
            
        return pred_id
        
    except Exception as e:
        # print(f"Prediction Error: {e}")
        return 0
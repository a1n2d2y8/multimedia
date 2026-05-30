### 前處理
```
python preprocess.py
```
把當前目錄下的HaGRIDv2_dataset_512裡面所有圖片處理成128X128的圖片與21個landmark，存到processed_data下，圖片依照類別存到各自的資料夾下，而landmark存在csv檔案中。

### 訓練
```
python train.py
```
dataset/GestureDataset: 把資料分為train和valid，並做data augmentation。
model/DualInputClassifier:  定義本次訓練使用的model
train.py: 做完dataset處理後，進行訓練，並選出按照本次project評分標準得到最高分的model作為最終的model，存在model_checkpoints中

### 測試
```
python test.py
```
使用testcase中的圖片做測試，testcase包含了5種目標類別以及15種N/A的類別各十張，總共200張圖片。
以上圖片是我自己拍攝以及從網路上下載的。
test會先把圖片做前處理成128x128 + landmarks，然後再使用inference.py中的predict進行測試。
測試結果會存到csv檔案test_result.csv中，128x128的圖片會存到debug資料夾下，使我們能夠確認圖片是否能被正確處理。

### pipeline
1. 下載處理好的processed_data放在根目錄下或是執行preprocess.py。
2. 執行train.py做訓練
3. 執行tes.py進行測試

### 其他
- 其他的model.py架構存在model資料夾下
- 測試時，inference.py裡面的DualInputClassifier必須要和訓練best model時的架構一樣。
- 每一次執行新的train.py，model_checkpoints裡面原有的model會被覆蓋掉。

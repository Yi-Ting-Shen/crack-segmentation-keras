# crack-segmentation-keras


### Deep learning framework
Tensorflow : 2.15.0  
Keras : 2.15.0

## 準備Dataset
這是一個語意分割任務，需要準備要訓練的圖片和對應的label。

訓練原圖是RGB，有3個channel的圖片  
<img src="./figure/trainingdata_img.png" width="400" alt="訓練原圖"/>  

對應的裂縫label是只有1個channel的圖片  
<img src="./figure/trainingdata_label.png" width="400" alt="訓練label"/>  

## 安裝環境   
```
pip install -r requirements.txt
```

## 訓練模型
正式進行訓練之前可以先修改 `config.py` 來設定全域變數。
```
python train.py
```
會印出訓練時每個Epoch的Loss和Dice Coefficient記錄。

每個Epoch的Loss  
<img src="./figure/history_loss.png" width="400" alt="history_loss"/>  

每個Epoch的Dice Coefficient  
<img src="./figure/history_dice.png" width="400" alt="history_dice"/>  




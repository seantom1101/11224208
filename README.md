# 專案實戰解析：基於深度學習建構卷積神經網路模型演算法，實現圖像辨識分類
11224208 張承均

## 📋 目錄

- [前言]
- [基礎知識介紹]
- [數據集收集]
- [模型訓練]
- [圖像辨識分類]
- [結果展示]
- [總結]
  
---

## 前言

隨著人工智慧的不斷發展，深度學習這門技術也越來越重要，許多人開啟了學習機器學習。本專案透過實戰案例，帶領大家從零開始設計實現一款基於深度學習的圖像辨識演算法。

**學習本計畫內容，你需要掌握以下基礎知識：**

1. Python 基礎語法
2. 計算機視覺庫 (OpenCV)
3. 深度學習框架 (TensorFlow)
4. 卷積神經網路 (CNN)

---

## 基礎知識介紹

### 1. Python
Python 是一個高層次的結合了解釋性、編譯性、互動性和物件導向的腳本語言。
- 學習連結：[Python學習](https://www.runoob.com/python3/python3-intro.html)

### 2. OpenCV
OpenCV 是一個開源的跨平台計算機視覺庫，實現了圖像處理和計算機視覺方面的許多通用演算法。
- 學習連結：[OpenCV學習](https://docs.opencv.org/4.x/index.html)

### 3. TensorFlow
TensorFlow 是Google開源的計算框架，可以很好地支援深度學習的各種演算法。
- 學習連結：[TensorFlow學習](https://tensorflow.google.cn/)

### 4. CNN (卷積神經網路)
卷積神經網路是一類包含卷積計算且具有深度結構的前饋神經網路，是深度學習的代表性演算法之一。
- 學習連結：[CNN學習](https://xie.infoq.cn/article/c4d846096c92c7dfcd6539075)

---

## 數據集收集
在進行影像辨識前，首先需要收集資料集，其次對於資料集做預處理，然後才能通過
深度卷積神經網路來進行特徵學習，得到估計分類模型。對於資料集的要求，在卷積神經網絡
（CNN）中，由於對輸入影像向量的權值參數的數量是固定的，所以在用卷積網路（CNN）對資料集
進行模型訓練前需進行影像預處理，確保輸入的影像尺寸是固定一致的。
<img width="536" height="45" alt="JP9DRjfC7CRtFiN8c_TjfLoWYoezGqym0WUiWeriqoexe8JmGJ2qV11IIg1feOiOAcXYWms7wVndFAjUeKFSIjvPZsT-puVcrBImOUMp6Ua_n8kNK5qX-PLfvhDA4YLLd39m5bddEUK5IcImQSg8Tw_3vP8E7dxxJN47LkDmsIF1WNKNT31djIeRbUbc1_K5tEp3vJrTztXX1AuBzfBileR" src="https://github.com/user-attachments/assets/eda0a3d3-0177-407f-871c-5431b6ab403a" />
圖一 分類網路模型流程圖

# 程式

**匯入套件**
```python
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.utils import to_categorical

print(f"TensorFlow Version: {tf.__version__}")
```



---

**定義核心類別**
```python
class SafeImageClassifier:
    def __init__(self, image_path, img_size=200):
        self.image_path = image_path
        self.img_size = img_size

        # 1. 取得所有資料夾名稱
        all_dirs = [
            d for d in os.listdir(image_path)
            if os.path.isdir(os.path.join(image_path, d))
        ]

        # 2. 過濾掉系統檔與預測用的資料夾
        self.classes = sorted([
            d for d in all_dirs
            if not d.startswith('.') and d != "predictPic"
        ])

        self.class_map = {c: i for i, c in enumerate(self.classes)}
        self.model = None
        self.df = None

        print(f"✅ 偵測到的訓練類別: {self.classes}")
        if "predictPic" in all_dirs:
            print("ℹ️ 已自動忽略 'predictPic' 資料夾，不將其視為訓練類別。")

    def resize_images(self):
        print("🔄 開始調整圖片大小...")
        for cls in self.classes:
            cls_folder = os.path.join(self.image_path, cls)
            for f in os.listdir(cls_folder):
                if f.startswith('.'): continue
                fp = os.path.join(cls_folder, f)
                try:
                    img = cv2.imread(fp)
                    if img is not None:
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        cv2.imwrite(fp, img)
                except:
                    pass
        print("✅ 圖片 Resize 完成")

    def generate_csv(self):
        data = []
        for cls in self.classes:
            cls_folder = os.path.join(self.image_path, cls)
            for f in os.listdir(cls_folder):
                if f.startswith('.'): continue
                data.append({
                    "path": os.path.join(cls_folder, f),
                    "label": self.class_map[cls]
                })
        self.df = pd.DataFrame(data)
        # 這裡不存檔也沒關係，直接存在記憶體中
        print(f"✅ 資料索引建立完成，共有 {len(self.df)} 張圖片")

    def build_model(self):
        inputs = Input(shape=(self.img_size, self.img_size, 3))

        # 第一層
        x = Conv2D(64, 3, activation='relu', padding='same')(inputs)
        x = MaxPooling2D()(x)
        x = BatchNormalization()(x)

        # 第二層
        x = Conv2D(64, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D()(x)
        x = BatchNormalization()(x)

        # 第三層
        x = Conv2D(32, 3, activation='relu', padding='same')(x)
        x = MaxPooling2D()(x)
        x = BatchNormalization()(x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)

        outputs = Dense(len(self.classes), activation='softmax')(x)

        self.model = Model(inputs, outputs)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("✅ 模型建構完成")

    def train(self, epochs=10, batch_size=2):
        if self.df is None or len(self.df) == 0:
            print("❌ 無圖片資料可訓練")
            return

        X, y = [], []
        for _, row in self.df.iterrows():
            try:
                img = cv2.imread(row['path'])
                if img is not None:
                    img = cv2.resize(img, (self.img_size, self.img_size))
                    img = img / 255.0
                    X.append(img)
                    y.append(row['label'])
            except:
                pass

        X = np.array(X)
        y = to_categorical(y, num_classes=len(self.classes))

        Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"🚀 開始訓練：訓練集 {len(Xtr)} 張, 驗證集 {len(Xva)} 張")

        self.model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch_size)

    def predict_image(self, img_path, name_map=None):
        if self.model is None:
            print("❌ 模型尚未訓練")
            return

        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ 找不到圖片: {img_path}")
            return

        img_display = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0
        img = np.expand_dims(img, 0)

        probs = self.model.predict(img)[0]
        idx = np.argmax(probs)
        cls = self.classes[idx]
        conf = float(probs[idx])

        label_show = name_map.get(cls, cls) if name_map else cls

        plt.figure(figsize=(4,4))
        plt.imshow(img_display)
        plt.title(f"{label_show} ({conf:.1%})")
        plt.axis('off')
        plt.show()

        return cls, conf
```
---

**初始化與訓練 (Training)**
```python
# 設定你的圖片資料夾路徑
PATH = "/content/drive/MyDrive/picture/"

# 1. 初始化
classifier = SafeImageClassifier(PATH)

# 2. 處理圖片與建立索引
classifier.resize_images()
classifier.generate_csv()

# 3. 建立模型 (已包含更多數據增強)
classifier.build_model()

# 4. 進行初步訓練
print("\n--- 階段一：初步訓練 (凍結基底模型) ---")
classifier.train(epochs=20, batch_size=32)
```

---

**預測 (Prediction)**
```python
# 設定你要預測的圖片路徑
test_img = "/content/drive/MyDrive/picture/cat/000001.jpg"

# 設定中文顯示對照表 (可選)
my_map = {
    "bird": "Bird (鳥)",
    "cat": "Cat (貓)"
}

# 執行預測
if os.path.exists(test_img):
    result, confidence = classifier.predict_image(test_img, name_map=my_map)
    print(f"預測結果: {result}, 信心度: {confidence:.4f}")
else:
    print(f"找不到測試圖片，請檢查路徑: {test_img}")
```

---

<img width="453" height="155" alt="image" src="https://github.com/user-attachments/assets/f75ebcc2-5083-4524-ad92-3b4794dc4fc9" />


**建立分類器:**

        classifier = SafeImageClassifier(image_path)

image_path 是資料集根目錄。會自動掃描資料夾，將每個子資料夾視為一個類別。初始化時會列出類別名稱。

**圖片處理:**

        classifier.resize_images()
        
將每個類別資料夾中的圖片 resize 成固定大小（200×200）。

避免訓練時因為尺寸不同造成錯誤。
        
        classifier.generate_csv()

生成 CSV，記錄每張圖片路徑與對應標籤。

CSV 方便後續讀取訓練資料。


---

# 結果圖





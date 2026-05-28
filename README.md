---

# CycleGAN Weather Translation (Sunny / Snowy / Cloudy)

## 專案簡介 / Overview

本專案實作 **CycleGAN** 進行非配對影像到影像轉換（unpaired image-to-image translation）。模型能夠學習在不同天氣場景中進行風格轉換（例如：**Sunny ↔ Snowy**），且不需要對應的（paired）影像資料。

### 中文介紹

CycleGAN 的核心在於 **Cycle-consistency loss**，它能在改變影像風格（如將晴天變雪地）的同時，完美保留原始場景的結構與物體輪廓。本專案生成器輸出層使用 `tanh`，因此輸入資料會自動縮放到 $[-1, 1]$ 以確保數值對齊。

### English Overview

This project applies CycleGAN for weather style transformation across scenes. CycleGAN does not require paired datasets; it preserves scene structure via cycle-consistency loss while changing textures and colors. The generator uses a `tanh` output layer, so inputs are scaled to $[-1, 1]$ accordingly.

---

## 專案重點 / Features

* **無需成對資料 (Unpaired Data):** 不需要同一地點在不同天氣下的對照圖。
* **結構保留 (Structure Preservation):** 透過 Cycle-consistency 確保轉換後場景不失真。
* **自動縮放 (Data Scaling):** 配合 `tanh` 激活函數，處理從 $[0, 1]$ 到 $[-1, 1]$ 的數值映射。
* **多域擴充:** 支援晴天 (Sunny)、雪地 (Snowy) 以及雲天 (Cloudy) 的轉換。

---

## 方法概述 / Methodology

### 1. 核心架構 / Architecture

CycleGAN 包含以下組件：

* **兩個生成器 (Generators):**
* $G_{X \rightarrow Y}$: 將晴天 ($X$) 轉換為雪地 ($Y$)。
* $G_{Y \rightarrow X}$: 將雪地 ($Y$) 轉換回晴天 ($X$)。


* **兩個判別器 (PatchGAN Discriminators):**
* $D_X$: 判別影像是否為真實的晴天。
* $D_Y$: 判別影像是否為真實的雪地。



### 2. 損失函數 / Loss Functions

* **Adversarial Loss:** 使用 **LSGAN (MSE Loss)** 以提高訓練穩定性。
* **Cycle-consistency Loss:** 使用 **L1 Loss** 來最小化 $G_{Y \rightarrow X}(G_{X \rightarrow Y}(X))$ 與原始 $X$ 之間的差距（預設權重 $\lambda = 10$）。

---

## 資料前處理 / Pre-processing

1. 使用 `torchvision.transforms.ToTensor()` 將像素值轉為 $[0, 1]$ 的 Float Tensor。
2. 由於 Generator 輸出層為 `tanh` ($[-1, 1]$)，訓練前需透過 `scale()` 函式將輸入線性縮放至 **$[-1, 1]$**。

---

## 專案結構 / Folder Structure

```text
.
├── data.py               # 資料讀取 (Dataloader) 與 scale() 處理
├── cyclegan_model.py     # 生成器與判別器架構定義
├── utils.py              # 視覺化工具 (imshow, save_samples)
├── train.py              # 訓練迴圈與 Loss 記錄
└── outputs/
    └── samples/           # 訓練過程中產生的對照影像

```

### 資料夾格式要求 / Dataset Format

本專案使用 `ImageFolder` 讀取，請確保每個資料夾內至少有一層子目錄：

```text
<DATA_DIR>/
├── sunny/
│   └── all/              # 存放所有晴天訓練圖片
├── snow/
│   └── all/              # 存放所有雪地訓練圖片
└── test_sunny/
    └── all/              # 存放測試用圖片

```

---

## 資料集來源 / Datasets

本專案建議使用以下 Kaggle 資料集：

1. [Weather Dataset](https://www.kaggle.com/datasets/jehanbhathena/weather-dataset)
2. [Rome Weather Classification](https://www.kaggle.com/datasets/rogeriovaz/rome-weather-classification)

---

## 參考與致謝 / Acknowledgements

* **Paper:** [CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593)
* **Data Credits:** 感謝上述 Kaggle 資料集的提供者。

---

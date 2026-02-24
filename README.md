# CycleGAN Weather Translation (Sunny / Snowy / Cloudy)

## 中文簡介
本專案使用 CycleGAN 進行非配對影像到影像轉換（unpaired image-to-image translation），學習在不同場景中改變天氣風格，例如 **Sunny ↔ Snowy**，並可延伸到 **Cloudy**。  
CycleGAN 不需要成對（paired）資料，透過 cycle-consistency loss 在轉換風格的同時保留場景結構；本專案的生成器輸出使用 `tanh`，因此輸入會縮放到 `[-1, 1]` 以對齊輸出範圍。

## English Overview
This project applies CycleGAN for **unpaired image-to-image translation** to transform weather styles across scenes, e.g., **Sunny ↔ Snowy**, and can be extended to **Cloudy**.  
CycleGAN does not require paired data; it preserves scene structure via cycle-consistency loss while changing style. The generator output uses `tanh`, so inputs are scaled to `[-1, 1]` accordingly.

---

## Features / 專案重點
- 不需要成對資料 / No paired data required  
- 以 cycle-consistency 保留場景內容 / Preserve structure via cycle-consistency  
- `tanh` 輸出搭配輸入縮放到 `[-1, 1]` / `tanh` output with input scaling to `[-1, 1]`

---

## Motivation / 動機
**中文：** 天氣/季節轉換常缺乏「同一地點不同天氣」的成對圖片；CycleGAN 能使用兩個 domain 的非成對資料訓練，並透過 cycle-consistency loss 讓轉換具可逆性（X→Y→X、Y→X→Y）。  
**English:** Paired images of the same location under different weather conditions are often unavailable. CycleGAN learns from unpaired domains and enforces reversibility via cycle-consistency (X→Y→X and Y→X→Y).

---

## Method Overview / 方法概述
CycleGAN 架構包含 / The CycleGAN architecture includes:
- **兩個生成器 / Two generators**
  - **G\_{X→Y}**: X（sunny）→ Y（snow）
  - **G\_{Y→X}**: Y（snow）→ X（sunny）
- **兩個判別器（PatchGAN）/ Two discriminators (PatchGAN)**
  - **D\_X**: 判別是否為真實 domain X / real vs fake in domain X
  - **D\_Y**: 判別是否為真實 domain Y / real vs fake in domain Y
- **Loss / 損失函數**
  - Adversarial loss：LSGAN（MSE）
  - Cycle-consistency：L1 loss × λ（預設 / default λ=10）

---

## Pre-processing / 前處理
- `torchvision.transforms.ToTensor()` 會將像素轉成 float tensor，範圍 **[0, 1]**  
  `ToTensor()` converts pixels to float tensors in **[0, 1]**
- 因 generator 最後使用 `tanh`（輸出 **[-1, 1]**），訓練前需把輸入從 **[0, 1]** 線性縮放到 **[-1, 1]**（本專案用 `scale()`）  
  Since the generator uses `tanh` (output **[-1, 1]**), inputs are linearly scaled from **[0, 1]** to **[-1, 1]** via `scale()`.

---

## Datasets / 資料集
本專案使用兩份 Kaggle 資料集（不直接上傳完整影像到 GitHub，請自行下載）  
We use two Kaggle datasets (images are not committed to GitHub; please download them yourself):

1) Weather Dataset  
- https://www.kaggle.com/datasets/jehanbhathena/weather-dataset

2) Rome Weather Classification  
- https://www.kaggle.com/datasets/rogeriovaz/rome-weather-classification

---

## Folder Structure / 專案結構
```text
.
├─ data.py               # dataloader + scale()
├─ cyclegan_model.py     # Generator / Discriminator / create_model()
├─ utils.py              # imshow / save_samples / view_samples
├─ train.py              # training loop + loss plot
└─ outputs/
   └─ samples/           # 訓練過程輸出的對照圖


Dataset Folder Format (Important) / 資料夾格式（重要）

本專案使用 torchvision.datasets.ImageFolder，因此每個 split 底下需要「至少一層」類別子資料夾（即使你不使用 label 也一樣）。
We use torchvision.datasets.ImageFolder, so each split must contain at least one class subfolder (even if labels are not used).

Two domains（sunny / snow）
<DATA_DIR>/
├─ sunny/
│  ├─ all/
│  │  └─ *.jpg
│  └─ test_sunny/
│     └─ all/
│        └─ *.jpg
└─ snow/
   ├─ all/
   │  └─ *.jpg
   └─ test_snow/
      └─ all/
         └─ *.jpg
Add a third domain（cloudy）
<DATA_DIR>/
└─ cloudy/
   ├─ all/
   │  └─ *.jpg
   └─ test_cloudy/
      └─ all/
         └─ *.jpg

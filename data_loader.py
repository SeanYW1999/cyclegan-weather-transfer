# loading in and transforming data
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# visualizing data
import matplotlib.pyplot as plt
import numpy as np
import warnings 

def get_data_loader(image_type, image_dir="請在執行時傳入你的資料夾路徑",
                    image_size=128, batch_size=16, num_workers=0):
    """Returns training and test data loaders for a given image type, either 'sunny' or 'snow'.
       These images will be resized to 128x128x3, by default, converted into Tensors, and normalized.
       
       注意：
       - transforms.ToTensor() 會把像素轉成 float tensor，並把範圍縮放到 [0, 1]
       - ImageFolder 需要「類別子資料夾」結構：root/sunny/<class_name>/*.jpg
         若 sunny 資料夾下直接放圖片而沒有子資料夾，ImageFolder 會讀不到類別
    """

    # resize and normalize the images
    # 圖片調整大小為 image_size；ToTensor() 轉為 PyTorch tensor，像素範圍會變成 [0, 1]
    transform = transforms.Compose([
        transforms.Resize(image_size),   # resize to image_size x image_size
        transforms.ToTensor()
    ])

    # get training and test directories
    image_path = image_dir  # 資料集根目錄（請改成你的資料集根目錄）
    train_path = os.path.join(image_path, image_type)  # 訓練資料路徑：根目錄/天氣類別（例如 sunny 或 snow）
    test_path  = os.path.join(image_path, f'test_{image_type}')  # 測試資料路徑：根目錄/test_天氣類別（例如 test_sunny）

    # define datasets using ImageFolder
    # ImageFolder 會依照「子資料夾名稱」自動產生 label；CycleGAN 訓練通常不需要 label，因此後面用 "_" 忽略
    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset  = datasets.ImageFolder(test_path, transform)

    # create and return DataLoaders
    # batch_size: 每次迭代取出多少張圖片
    # shuffle=True: 訓練資料打亂，減少模型記住順序的風險
    # num_workers: 資料載入的平行 worker 數；Windows/Notebook 有時用 0 較穩
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


# Create train and test dataloaders for images from the two domains X and Y
# 這裡把 domain X 設為 sunny、domain Y 設為 snow（之後 CycleGAN 會學 X↔Y 的轉換）
# 若要避免函數預設值寫死路徑，可在此傳入 image_dir="你的資料集根目錄"
dataloader_X, test_dataloader_X = get_data_loader(image_type='sunny')
dataloader_Y, test_dataloader_Y = get_data_loader(image_type='snow')


# helper imshow function
# make_grid 產生的 tensor 形狀是 (C, H, W)，matplotlib 需要 (H, W, C)，所以要 transpose
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")  # 顯示圖片時通常把座標軸關掉較乾淨


# get some images from X
# dataloader 迭代後會回傳 (images, labels)
dataiter = iter(dataloader_X)

# "_" 是用來忽略 label（CycleGAN 不用分類標籤）
images, _ = next(dataiter)

# show images
# make_grid 會把一個 batch 拼成網格圖，方便快速檢查資料
fig = plt.figure(figsize=(12, 8))
imshow(torchvision.utils.make_grid(images))
plt.show()


# current range
# 取 batch 的第一張圖片，檢查像素範圍；ToTensor 後通常是 [0, 1]
img = images[0]
print('Min: ', img.min())
print('Max: ', img.max())


# helper scale function
def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1.

       修正說明：
       - 這裡的 x 是 ToTensor() 後的 tensor，通常已經在 [0, 1]（不是 0-255）
       - 目的：把 [0, 1] 線性映射到 [-1, 1]，與 generator 最後的 tanh 輸出範圍一致
    '''
    
    # scale from 0-1 to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x


# scaled range
# 縮放後的範圍應接近 [-1, 1]
scaled_img = scale(img)
print('Scaled min: ', scaled_img.min())
print('Scaled max: ', scaled_img.max())

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

def get_data_loader(image_type, image_dir="請在執行時傳入你的資料夾路徑，例如：D:/work/weather/",
                    image_size=128, batch_size=16, num_workers=0):
    """Returns training and test data loaders for a given image type, either 'sunny' or 'snow'.
       These images will be resized to 128x128x3, by default, converted into Tensors, and normalized.
    """

    # resize and normalize the images
    # 圖片調整大小為 image_size，transforms.ToTensor()將圖片轉為 pytorch 張量
    transform = transforms.Compose([transforms.Resize(image_size), # resize to 128x128
                                    transforms.ToTensor()])

    # get training and test directories
    image_path =  image_dir # 主圖像（請改成你的資料集根目錄）
    train_path = os.path.join(image_path, image_type) # 訓練數據集路徑：根目錄/天氣類別資料夾（例如 sunny 或 snow）
    test_path = os.path.join(image_path, 'test_{}'.format(image_type)) # 測試數據集路徑：根目錄/test_天氣類別資料夾（例如 test_sunny）

    # define datasets using ImageFolder
    # 使用 PyTorch 的 ImageFolder 類創建訓練和測試數據集。從指定路徑(train_path or test_path)加載圖片，並根據目錄結構將它們分配到不同的類別。
    train_dataset = datasets.ImageFolder(train_path, transform)
    test_dataset = datasets.ImageFolder(test_path, transform)

    # create and return DataLoaders
    # batch_size: 每次訓練時，一次處理16圖片; shuffle: ==Ture時，圖片順序隨機排列，避免過擬合，test順序不變; num_workers=num_workers: 當num_workers==0使用單線程，>0使用多線程，這裡是用0
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader

# Create train and test dataloaders for images from the two domains X and Y
# image_type = directory names for our data
dataloader_X, test_dataloader_X = get_data_loader(image_type='sunny')
dataloader_Y, test_dataloader_Y = get_data_loader(image_type='snow')

# helper imshow function
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

# get some images from X
dataiter = iter(dataloader_X)
# the "_" is a placeholder for no labels
images, _ = next(dataiter)

# show images
fig = plt.figure(figsize=(12, 8))
imshow(torchvision.utils.make_grid(images))

# current range
img = images[0]

print('Min: ', img.min())
print('Max: ', img.max())

# helper scale function
def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-255.'''
    
    # scale from 0-1 to feature_range
    min, max = feature_range
    x = x * (max - min) + min
    return x

# scaled range
scaled_img = scale(img)

print('Scaled min: ', scaled_img.min())
print('Scaled max: ', scaled_img.max())

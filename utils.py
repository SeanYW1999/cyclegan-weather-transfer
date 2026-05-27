import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

def view_samples(iteration, sample_dir):
    """Load in and display saved sample images.
    
    目的：
    - 讀取 save_samples() 存下來的對照圖，並用 matplotlib 顯示
    
    參數：
    - iteration：對應 epoch 編號（例如 500、1000）
    - sample_dir：存放 sample 圖片的資料夾（例如 outputs/samples）
    """
    # 嘗試載入圖片
    try:
        x2y_path = os.path.join(sample_dir, f'epoch_{iteration}_comparison_Y.png')
        y2x_path = os.path.join(sample_dir, f'epoch_{iteration}_comparison_X.png')

        if not os.path.exists(x2y_path) or not os.path.exists(y2x_path):
            raise FileNotFoundError

        x2y = plt.imread(x2y_path)
        y2x = plt.imread(y2x_path)

    except FileNotFoundError:
        print(f"Error: Samples for iteration {iteration} not found in '{sample_dir}'.")
        return  # 中止函數執行

    # 顯示圖片：上面放 X->Y，下面放 Y->X
    fig, (ax1, ax2) = plt.subplots(figsize=(18, 20), nrows=2, ncols=1, sharey=True, sharex=True)
    ax1.imshow(x2y)
    ax1.set_title('X to Y')
    ax1.axis("off")

    ax2.imshow(y2x)
    ax2.set_title('Y to X')
    ax2.axis("off")

    plt.show()

## utils
n_epochs = 2000 # keep this small when testing if a model first works, then increase it to >=1000
losses = training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y, n_epochs=n_epochs)

# view samples at iteration 100
view_samples(100, 'samples_cyclegan')

import os
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np


# helper imshow function
# 將 tensor 格式的影像顯示在 matplotlib 上
# make_grid 產生的影像格式是 (C, H, W)，matplotlib 需要 (H, W, C)，所以要轉置
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")  # 顯示圖片時通常不需要座標軸


def save_samples(epoch, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=16, output_dir=''):
    """Saves generated samples along with the original images for comparison.
    
    目的：
    - 固定一組 X/Y 的測試圖片（fixed_X / fixed_Y），每隔幾個 epoch 存一次輸出結果
    - 方便觀察 CycleGAN 訓練過程：生成結果是否越來越像目標 domain
    
    輸出內容：
    - epoch_{epoch}_comparison_X.png：左半是真實 Y（例如 snow），右半是 Y->X 生成結果（例如 sunny）
    - epoch_{epoch}_comparison_Y.png：左半是真實 X（例如 sunny），右半是 X->Y 生成結果（例如 snow）
    
    注意：
    - fixed_X / fixed_Y 應該已經被 scale 到 [-1, 1]，與 generator tanh 輸出一致
    - normalize=True 會把 tensor 的值映射到 [0, 1] 再存成圖片
    """
    # 創建輸出目錄（若 output_dir=''，會存到目前工作目錄）
    os.makedirs(output_dir, exist_ok=True)

    # 確保模型和數據在同一設備上（CPU 或 GPU）
    device = next(G_YtoX.parameters()).device
    fixed_Y = fixed_Y.to(device)
    fixed_X = fixed_X.to(device)

    # 切到 eval 模式：BatchNorm 會使用推論統計，結果更穩定
    G_YtoX.eval()
    G_XtoY.eval()

    with torch.no_grad():
        # 生成圖片
        # fake_X：把 domain Y 的圖片轉成 X（例如 snow -> sunny）
        # fake_Y：把 domain X 的圖片轉成 Y（例如 sunny -> snow）
        fake_X = G_YtoX(fixed_Y[:batch_size]).detach()
        fake_Y = G_XtoY(fixed_X[:batch_size]).detach()

    # 將原始圖片與生成圖片沿寬度拼接，形成「左原圖 / 右生成圖」對照
    # 拼接 Y->X
    comparison_X = torch.cat((fixed_Y[:batch_size], fake_X), dim=3)  # dim=3 表示沿 width 拼接
    # 拼接 X->Y
    comparison_Y = torch.cat((fixed_X[:batch_size], fake_Y), dim=3)

    # 保存拼接後的圖片
    file_path_X = os.path.join(output_dir, f'epoch_{epoch}_comparison_X.png')
    file_path_Y = os.path.join(output_dir, f'epoch_{epoch}_comparison_Y.png')

    torchvision.utils.save_image(comparison_X, file_path_X, normalize=True)
    torchvision.utils.save_image(comparison_Y, file_path_Y, normalize=True)

    print(f"Comparison samples saved at epoch {epoch}!")


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

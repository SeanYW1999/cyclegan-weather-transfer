import os
import argparse
import torch
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from data_loader import get_data_loader, scale
from cyclegan_model import create_model
from utils import save_samples

# -----------------------------
# Loss functions
# -----------------------------
def real_mse_loss(D_out):
    # 判別器輸出越接近 1 越好（真樣本目標=1）
    # L_real = mean((D(x) - 1)^2)
    return torch.mean((D_out - 1) ** 2)

def fake_mse_loss(D_out):
    # 判別器輸出越接近 0 越好（假樣本目標=0）
    # L_fake = mean((D(G(z)) - 0)^2) = mean(D(G(z))^2)
    return torch.mean(D_out ** 2)

def cycle_consistency_loss(real_im, reconstructed_im, lambda_weight):
    # Cycle consistency：希望 x -> G(x) -> F(G(x)) 可以回到原來的 x
    # 使用 L1 loss（abs）並乘上權重 lambda_weight
    return torch.mean(torch.abs(real_im - reconstructed_im)) * lambda_weight

# -----------------------------
# Optimizers
# -----------------------------

import torch.optim as optim

# hyperparams for Adam optimizers
lr= 0.0002
beta1= 0.5
beta2= 0.999

g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())  # Get generator parameters

# Create optimizers for the generators and discriminators
g_optimizer = optim.Adam(g_params, lr=lr, betas=(beta1, beta2))
d_x_optimizer = optim.Adam(D_X.parameters(), lr=lr, betas=(beta1, beta2))
d_y_optimizer = optim.Adam(D_Y.parameters(), lr=lr, betas=(beta1, beta2))

# -----------------------------
# Save result images
# -----------------------------
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
    
# -----------------------------
# Training loop
# -----------------------------
def training_loop(dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y,
                  G_XtoY, G_YtoX, D_X, D_Y,
                  g_optimizer, d_x_optimizer, d_y_optimizer,
                  n_epochs=1000,
                  print_every=10,
                  sample_every=500,
                  lambda_cycle=10,
                  out_dir="outputs/samples"):
    """
    CycleGAN 訓練迴圈（sunny <-> snow）
    - 先訓練判別器 D_X / D_Y：分辨真圖與生成圖
    - 再訓練生成器 G_XtoY / G_YtoX：騙過判別器 + 維持 cycle consistency

    注意：
    - 你的資料在 data.py 會先是 [0,1]，此處用 scale() 轉到 [-1,1] 以符合 generator tanh 輸出範圍
    - D 的 loss 使用 MSE（LSGAN 形式）
    - cycle consistency 使用 L1 loss 並乘 lambda_cycle
    """

    losses = []

    # 固定測試樣本：每隔 sample_every 儲存一次結果，用來觀察模型進步
    test_iter_X = iter(test_dataloader_X)
    test_iter_Y = iter(test_dataloader_Y)

    fixed_X = next(test_iter_X)[0]
    fixed_Y = next(test_iter_Y)[0]
    fixed_X = scale(fixed_X)
    fixed_Y = scale(fixed_Y)

    # 以兩個 dataloader 中較小的長度作為每個 epoch 的可用 batches
    iter_X = iter(dataloader_X)
    iter_Y = iter(dataloader_Y)
    batches_per_epoch = min(len(iter_X), len(iter_Y))

    # device 統一管理，避免每次迭代重建
    device = next(G_XtoY.parameters()).device

    for epoch in range(1, n_epochs + 1):
        G_XtoY.train()
        G_YtoX.train()
        D_X.train()
        D_Y.train()
        
        # 重新建立 iterator（避免 StopIteration）
        if epoch % batches_per_epoch == 0:
            iter_X = iter(dataloader_X)
            iter_Y = iter(dataloader_Y)

        images_X, _ = next(iter_X)
        images_Y, _ = next(iter_Y)

        # 將 [0,1] 映射到 [-1,1]
        images_X = scale(images_X)
        images_Y = scale(images_Y)

        images_X = images_X.to(device)
        images_Y = images_Y.to(device)

        # ============================================
        #            TRAIN THE DISCRIMINATORS
        # ============================================

        # ----- D_X -----
        d_x_optimizer.zero_grad()

        # 真 X 圖的 loss
        D_X_real_loss = real_mse_loss(D_X(images_X))

        # 生成假 X 圖（由真 Y 圖經 G_YtoX 生成）
        G_Y2X_fake_image = G_YtoX(images_Y)

        # 假 X 圖的 loss
        D_X_fake_loss = fake_mse_loss(D_X(G_Y2X_fake_image.detach()))

        # D_X total loss
        d_x_loss = D_X_real_loss + D_X_fake_loss
        d_x_loss.backward()
        d_x_optimizer.step()

        # ----- D_Y -----
        d_y_optimizer.zero_grad()

        D_Y_real_loss = real_mse_loss(D_Y(images_Y))
        G_X2Y_fake_image = G_XtoY(images_X)
        D_Y_fake_loss = fake_mse_loss(D_Y(G_X2Y_fake_image.detach()))

        d_y_loss = D_Y_real_loss + D_Y_fake_loss
        d_y_loss.backward()
        d_y_optimizer.step()

        # =========================================
        #            TRAIN THE GENERATORS
        # =========================================
        g_optimizer.zero_grad()

        # Y -> X
        G_X_img = G_YtoX(images_Y)
        G_X_real_loss = real_mse_loss(D_X(G_X_img))

        # cycle: Y -> X -> Y
        G_Y_reconstructed = G_XtoY(G_X_img)
        G_Y_consistency_loss = cycle_consistency_loss(images_Y, G_Y_reconstructed, lambda_cycle)

        # X -> Y
        G_Y_img = G_XtoY(images_X)
        G_Y_real_loss = real_mse_loss(D_Y(G_Y_img))

        # cycle: X -> Y -> X
        G_X_reconstructed = G_YtoX(G_Y_img)
        G_X_consistency_loss = cycle_consistency_loss(images_X, G_X_reconstructed, lambda_cycle)

        # generator total loss
        g_total_loss = G_X_real_loss + G_Y_real_loss + G_Y_consistency_loss + G_X_consistency_loss
        g_total_loss.backward()
        g_optimizer.step()

        # log
        if epoch % print_every == 0:
            losses.append((d_x_loss.item(), d_y_loss.item(), g_total_loss.item()))
            print(
                "Epoch [{:5d}/{:5d}] | d_X_loss: {:6.4f} | d_Y_loss: {:6.4f} | g_total_loss: {:6.4f}".format(
                    epoch, n_epochs, d_x_loss.item(), d_y_loss.item(), g_total_loss.item()
                )
            )

        # save samples
        if epoch % sample_every == 0:
            save_samples(epoch, fixed_Y, fixed_X, G_YtoX, G_XtoY, batch_size=16, output_dir=out_dir)

    return losses

# 繪製 loss 曲線（d_X / d_Y / generator）
fig, ax = plt.subplots(figsize=(12, 8))
losses = np.array(losses)
plt.plot(losses.T[0], label="Discriminator, X", alpha=0.5)
plt.plot(losses.T[1], label="Discriminator, Y", alpha=0.5)
plt.plot(losses.T[2], label="Generators", alpha=0.5)
plt.title("Training Losses")
plt.legend()
    plt.show()

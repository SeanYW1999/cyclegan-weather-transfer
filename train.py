import os
import argparse
import torch
import torch.optim as optim
import torchvision

import matplotlib.pyplot as plt
import numpy as np

from data_loader import get_data_loader, scale
from cyclegan_model import create_model


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


# -----------------------------
# CLI / main
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="資料集根目錄，例如 D:/work/weather/")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--beta1", type=float, default=0.5)
    p.add_argument("--beta2", type=float, default=0.999)
    p.add_argument("--n_res_blocks", type=int, default=6)
    p.add_argument("--sample_every", type=int, default=500)
    p.add_argument("--print_every", type=int, default=10)
    p.add_argument("--lambda_cycle", type=float, default=10.0)
    p.add_argument("--out_dir", type=str, default="outputs/samples")
    p.add_argument("--num_workers", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 建立資料
    dataloader_X, test_dataloader_X = get_data_loader(
        image_type="sunny",
        image_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    dataloader_Y, test_dataloader_Y = get_data_loader(
        image_type="snow",
        image_dir=args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 建立模型
    G_XtoY, G_YtoX, D_X, D_Y = create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=args.n_res_blocks)

    # 建立 optimizer（Adam 超參數是 CycleGAN/LSGAN 常見設定）
    g_params = list(G_XtoY.parameters()) + list(G_YtoX.parameters())
    g_optimizer = optim.Adam(g_params, args.lr, [args.beta1, args.beta2])
    d_x_optimizer = optim.Adam(D_X.parameters(), args.lr, [args.beta1, args.beta2])
    d_y_optimizer = optim.Adam(D_Y.parameters(), args.lr, [args.beta1, args.beta2])

    # 開始訓練
    losses = training_loop(
        dataloader_X, dataloader_Y, test_dataloader_X, test_dataloader_Y,
        G_XtoY, G_YtoX, D_X, D_Y,
        g_optimizer, d_x_optimizer, d_y_optimizer,
        n_epochs=args.epochs,
        print_every=args.print_every,
        sample_every=args.sample_every,
        lambda_cycle=args.lambda_cycle,
        out_dir=args.out_dir
    )

    # 繪製 loss 曲線（d_X / d_Y / generator）
    fig, ax = plt.subplots(figsize=(12, 8))
    losses = np.array(losses)
    plt.plot(losses.T[0], label="Discriminator, X", alpha=0.5)
    plt.plot(losses.T[1], label="Discriminator, Y", alpha=0.5)
    plt.plot(losses.T[2], label="Generators", alpha=0.5)
    plt.title("Training Losses")
    plt.legend()
    plt.show()

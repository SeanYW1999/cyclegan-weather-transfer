# 包含：卷積/反卷積輔助函數、Discriminator、ResidualBlock、CycleGenerator、create_model

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a convolutional layer, with optional batch normalization.
       建立 2D 卷積層，可選擇是否加入 BatchNorm2d。
       
       參數說明：
       - in_channels / out_channels：輸入/輸出通道數
       - kernel_size：卷積核大小
       - stride：步長（預設 2，會做下採樣）
       - padding：補零（預設 1）
       - batch_norm：是否加入批次標準化（通常除了第一層或某些特殊層以外會開）
       
       回傳：
       - nn.Sequential(...)：包含 Conv2d（以及可選的 BatchNorm2d）
    """
    layers = []
    # 建立二維卷積層（bias=False：搭配 BatchNorm 時通常不需要 bias）
    conv_layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    layers.append(conv_layer)

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))  # 批次標準化層，穩定訓練、加速收斂
    return nn.Sequential(*layers)  # 將多層打包成一個模組


class Discriminator(nn.Module):
    """
    判別器（PatchGAN 風格輸出）
    輸入：RGB 影像 (B, 3, H, W)
    輸出：一張「真偽分數圖」(B, 1, h, w)，不是單一 scalar。
         這種作法常見於 CycleGAN / Pix2Pix，可讓模型更注重局部紋理與風格。
    """

    def __init__(self, conv_dim=64):
        super(Discriminator, self).__init__()  # 初始化父類

        # Define all convolutional layers
        # Should accept an RGB image as input and output a single value
        # 注意：註解寫 single value，但實際輸出是 1-channel feature map（PatchGAN）
        self.conv1 = conv(3, 64, 4, batch_norm=False)   # 第一層通常不做 BN
        self.conv2 = conv(64, 128, 4)                   # 後續層使用 BN，穩定分佈
        self.conv3 = conv(128, 256, 4)
        self.conv4 = conv(256, 512, 4)

        # 最後一層 stride=1：不再下採樣，只做最後分類輸出（1 channel）
        self.conv5 = conv(512, 1, 4, 1, batch_norm=False)

    # 使用 LeakyReLU：判別器常用，避免 ReLU 導致「死神經元」
    def forward(self, x):
        # define feedforward behavior
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = self.conv5(x)

        return x


# residual block class
# x -> [Conv Layer] -> [ReLU] -> [Conv Layer] -> + -> y
# x: 殘差塊輸入
# F(x): 殘差函數（兩層 conv + activation）
# H(x)=x+F(x): 殘差連接（skip connection），讓深層網路更容易訓練、保留內容結構
class ResidualBlock(nn.Module):
    """Defines a residual block.
       This adds an input x to a convolutional layer (applied to x) with the same size input and output.
       These blocks allow a model to learn an effective transformation from one domain to another.

       殘差塊的輸入與輸出維度相同（conv_dim 不變），可堆疊多個以提升生成器表達能力。
    """
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        # conv_dim = number of inputs / outputs channels in this block

        # define two convolutional layers + batch normalization that will act as our residual function, F(x)
        # layers should have the same shape input as output; kernel_size=3 通常用於保留空間大小
        self.conv1 = conv(conv_dim, conv_dim, 3, 1)  # stride=1：不改變空間尺寸
        self.conv2 = conv(conv_dim, conv_dim, 3, 1)

    def forward(self, x):
        # apply activation after first conv
        input_x = x  # 保留原輸入做 skip connection
        x = F.leaky_relu(self.conv1(x))
        # x + F(x)
        x = input_x + self.conv2(x)
        return x


def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
    """Creates a transpose convolutional layer, with optional batch normalization.
       建立 2D 轉置卷積層（反卷積），常用於上採樣/還原空間尺寸。
       
       參數說明同 conv()，差別在於使用 ConvTranspose2d。
    """
    layers = []
    # 建立二維轉置卷積層
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False))
    # optional batch norm layer
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class CycleGenerator(nn.Module):
    """
    CycleGAN 生成器（Encoder -> ResNet blocks -> Decoder）
    輸入：RGB 影像 (B, 3, H, W)；通常 H=W=128
    輸出：RGB 影像 (B, 3, H, W)，最後用 tanh 將像素限制在 [-1, 1]

    結構概念：
    - Encoder：多層 stride=2 conv，下採樣並提取語意特徵
    - ResNet：多個 residual blocks，學習 domain style 的轉換（例如 sunny <-> snow）
    - Decoder：多層 transpose conv，上採樣回原尺寸並生成輸出影像
    """

    # conv_dim: 基礎通道數, n_res_blocks: 殘差塊的數量
    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()  # 初始化父類

        # 1. Define the encoder part of the generator
        # 連續下採樣 3 次：空間尺寸約變成 1/8（例如 128 -> 16）
        self.enc_conv1 = conv(3, conv_dim, 4, 2)                 # 3 -> 64
        self.enc_conv2 = conv(conv_dim, conv_dim*2, 4, 2)        # 64 -> 128
        self.enc_conv3 = conv(conv_dim*2, conv_dim*4, 4, 2)      # 128 -> 256

        # 2. Define the resnet part of the generator
        # 堆疊 n_res_blocks 個 residual blocks（通道數維持 conv_dim*4）
        l = [ResidualBlock(conv_dim*4) for i in range(n_res_blocks)]
        self.resBlock = nn.Sequential(*l)

        # 3. Define the decoder part of the generator
        # 上採樣 3 次：尺寸回到原本大小
        self.dec_conv1 = deconv(conv_dim*4, conv_dim*2, 4, 2)
        self.dec_conv2 = deconv(conv_dim*2, conv_dim, 4, 2)
        self.dec_conv3 = deconv(conv_dim, 3, 4, 2)

    def forward(self, x):
        """Given an image x, returns a transformed image.
           給定輸入影像 x，輸出轉換後影像（同尺寸）。
        """
        # encoder：提取特徵 + 下採樣
        x = F.leaky_relu(self.enc_conv1(x))
        x = F.leaky_relu(self.enc_conv2(x))
        x = F.leaky_relu(self.enc_conv3(x))

        # resnet blocks：在壓縮特徵空間做風格轉換
        x = self.resBlock(x)

        # decoder：上採樣回原尺寸
        x = F.leaky_relu(self.dec_conv1(x))
        x = F.leaky_relu(self.dec_conv2(x))

        # tanh：輸出像素壓到 [-1, 1]，需配合資料前處理（將輸入也 scale 到 [-1, 1]）
        x = F.tanh(self.dec_conv3(x))
        return x


def create_model(g_conv_dim=64, d_conv_dim=64, n_res_blocks=6):
    """Builds the generators and discriminators.
       建立 CycleGAN 的兩個生成器與兩個判別器：
       - G_XtoY：X -> Y（例如 sunny -> snow）
       - G_YtoX：Y -> X（例如 snow -> sunny）
       - D_X：判別 X domain 真假
       - D_Y：判別 Y domain 真假

       若 CUDA 可用，會自動把模型移到 GPU。
    """

    # Instantiate generators
    G_XtoY = CycleGenerator(g_conv_dim, n_res_blocks)
    G_YtoX = CycleGenerator(g_conv_dim, n_res_blocks)

    # Instantiate discriminators
    D_X = Discriminator(d_conv_dim)
    D_Y = Discriminator(d_conv_dim)

    # move models to GPU, if available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        G_XtoY.to(device)
        G_YtoX.to(device)
        D_X.to(device)
        D_Y.to(device)
        print('Models moved to GPU.')
    else:
        print('Only CPU available.')

    return G_XtoY, G_YtoX, D_X, D_Y

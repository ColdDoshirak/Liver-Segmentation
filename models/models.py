"""
Модели для сегментации печени на КТ изображениях.
Включает реализации различных архитектур:
1. U-Net (базовая версия с улучшениями)
2. Attention U-Net (с attention механизмами)
3. 3D U-Net (для объемной сегментации)
4. ResUNet (с residual connections)
5. Swin UNETR (трансформерная архитектура)
6. nnU-Net (адаптивная архитектура)
7. UNet++ (вложенные плотные соединения)
8. EfficientNet-UNet (с предобученным энкодером)
9. TransUNet (гибрид CNN-Transformer)
10. CoTr (Convolutional Transformer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Union
from abc import ABC, abstractmethod


class BaseSegmentationModel(nn.Module, ABC):
    """
    Базовый класс для всех моделей сегментации печени.
    Определяет единый интерфейс и общие методы.
    """
    def __init__(self, n_channels: int, n_classes: int, **kwargs):
        super(BaseSegmentationModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.model_name = self.__class__.__name__
        
    @abstractmethod
    def forward(self, x):
        """Прямой проход модели"""
        pass
    
    def get_model_info(self) -> dict:
        """
        Получение информации о модели
        
        Returns:
            dict: Информация о модели
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'input_channels': self.n_channels,
            'output_classes': self.n_classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2)  # Assuming float32
        }
    
    def save_checkpoint(self, filepath: str, optimizer=None, epoch=None, metrics=None):
        """
        Сохранение чекпоинта модели
        
        Args:
            filepath: Путь для сохранения
            optimizer: Оптимизатор
            epoch: Номер эпохи
            metrics: Метрики
        """
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        if epoch is not None:
            checkpoint['epoch'] = epoch
            
        if metrics is not None:
            checkpoint['metrics'] = metrics
            
        torch.save(checkpoint, filepath)
    
    @classmethod
    def load_checkpoint(cls, filepath, device=None):
        """
        Загрузка чекпоинта модели
        
        Args:
            filepath: Путь к чекпоинту
            device: Устройство
            
        Returns:
            Модель с загруженными весами
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        checkpoint = torch.load(filepath, map_location=device)
        
        # Создание экземпляра модели
        model = cls(
            n_channels=checkpoint['n_channels'],
            n_classes=checkpoint['n_classes']
        )
        
        # Загрузка весов
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        return model, checkpoint


class DoubleConv(nn.Module):
    """Двойная свертка с BatchNorm и ReLU"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling с maxpool затем double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling затем double conv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Выходной слой свертки"""
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(BaseSegmentationModel):
    """
    U-Net архитектура для сегментации печени
    """
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False):
        super(UNet, self).__init__(n_channels, n_classes)
        self.bilinear = bilinear

        # Архитектура, соответствующая файлу весов UNet_best.pth
        self.inc = ConvBlock(n_channels, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(256, 512)
        )
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(512, 1024)
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = ConvBlock(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = ConvBlock(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = ConvBlock(128, 64)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)
        x = torch.cat([x3, x], dim=1)
        x = self.conv2(x)
        
        x = self.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.conv3(x)
        
        x = self.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.conv4(x)
        
        logits = self.outc(x)
        return logits


class ConvBlock(nn.Module):
    """
    Блок свертки, соответствующий архитектуре в файле весов
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SimpleConvBlock(nn.Module):
    """
    Простой блок свертки без BatchNorm, соответствующий архитектуре в файле весов
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    """
    Attention блок для Attention U-Net
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUp(nn.Module):
    """
    Upsampling блок с attention механизмом
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super(AttentionUp, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
            
        self.attention = AttentionBlock(F_g=in_channels // 2, F_l=in_channels // 2, F_int=in_channels // 4)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x2 = self.attention(g=x1, x=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class AttentionUNet(BaseSegmentationModel):
    """
    Attention U-Net архитектура для сегментации печени
    """
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False):
        super(AttentionUNet, self).__init__(n_channels, n_classes)
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = AttentionUp(1024, 512 // factor, bilinear)
        self.up2 = AttentionUp(512, 256 // factor, bilinear)
        self.up3 = AttentionUp(256, 128 // factor, bilinear)
        self.up4 = AttentionUp(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv3D(nn.Module):
    """3D двойная свертка с BatchNorm и ReLU"""
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """3D Downscaling с maxpool затем double conv"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """3D Upscaling затем double conv"""
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    """3D выходной слой свертки"""
    def __init__(self, in_channels: int, out_channels: int):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(BaseSegmentationModel):
    """
    3D U-Net архитектура для объемной сегментации печени
    """
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False):
        super(UNet3D, self).__init__(n_channels, n_classes)
        self.bilinear = bilinear

        self.inc = DoubleConv3D(n_channels, 64)
        self.down1 = Down3D(64, 128)
        self.down2 = Down3D(128, 256)
        self.down3 = Down3D(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down3D(512, 1024 // factor)
        self.up1 = Up3D(1024, 512 // factor, bilinear)
        self.up2 = Up3D(512, 256 // factor, bilinear)
        self.up3 = Up3D(256, 128 // factor, bilinear)
        self.up4 = Up3D(128, 64, bilinear)
        self.outc = OutConv3D(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class ResidualBlock(nn.Module):
    """
    Residual блок для ResUNet
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResUNet(BaseSegmentationModel):
    """
    ResUNet архитектура с residual connections
    """
    def __init__(self, n_channels: int, n_classes: int):
        super(ResUNet, self).__init__(n_channels, n_classes)
        
        # Encoder
        self.enc1 = ResidualBlock(n_channels, 64)
        self.enc2 = ResidualBlock(64, 128, stride=2)
        self.enc3 = ResidualBlock(128, 256, stride=2)
        self.enc4 = ResidualBlock(256, 512, stride=2)
        
        # Center
        self.center = ResidualBlock(512, 1024, stride=2)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(1024, 512)
        
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(512, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(256, 128)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(128, 64)
        
        # Output
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # Center
        center = self.center(e4)
        
        # Decoder
        d4 = self.up4(center)
        d4 = torch.cat([e4, d4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([e3, d3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([e2, d2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([e1, d1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        logits = self.outc(d1)
        return logits


class WindowAttention(nn.Module):
    """
    Window-based multi-head self attention (W-MSA) модуль для Swin UNETR
    """
    def __init__(self, dim: int, window_size: Tuple[int, int], num_heads: int, qkv_bias: bool = True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

        trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    """
    MLP модуль для Swin UNETR
    """
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: Tuple[int, int]) -> torch.Tensor:
    """
    Разделение изображения на окна
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: Tuple[int, int], H: int, W: int) -> torch.Tensor:
    """
    Восстановление изображения из окон
    """
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer блок для Swin UNETR
    """
    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int, int] = (7, 7), 
                 shift_size: Tuple[int, int] = (0, 0), mlp_ratio: float = 4.0, qkv_bias: bool = True, 
                 drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, window_size=window_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x, mask_matrix: Optional[torch.Tensor] = None):
        B, H, W, C = x.shape
        x = x.view(B, H, W, C)

        # Циклический сдвиг
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        else:
            shifted_x = x

        # Разделение на окна
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)

        # Attention
        attn_windows = self.attn(x_windows, mask=mask_matrix)

        # Объединение окон
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Обратный циклический сдвиг
        if self.shift_size[0] > 0 or self.shift_size[1] > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)

        # FFN
        x = x + self.mlp(self.norm2(x))
        return x


class PatchMerging(nn.Module):
    """
    Patch Merging слой для Swin UNETR
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape
        x = x.contiguous()
        x = x.view(B, H // 2, 2, W // 2, 2, C)
        x = x.permute(0, 1, 3, 4, 2, 5).contiguous()
        x = x.view(B, H // 2, W // 2, 4 * C)
        x = self.norm(x)
        x = x.view(B, -1, 4 * C)
        x = self.reduction(x)
        return x


class BasicLayer(nn.Module):
    """
    Basic Swin Transformer слой для Swin UNETR
    """
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: Tuple[int, int] = (7, 7),
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2)
        
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=(0, 0) if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop
            )
            for i in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class PatchEmbed(nn.Module):
    """
    Image to Patch Embedding для Swin UNETR
    """
    def __init__(self, img_size: Tuple[int, int] = (224, 224), patch_size: Tuple[int, int] = (4, 4), 
                 in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        img_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        patch_size = (patch_size[0], patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = img_size[0] * img_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """
    Truncated normal initialization
    """
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(min=a, max=b)
    return tensor


class SwinUNETR(BaseSegmentationModel):
    """
    Swin UNETR архитектура на основе трансформеров
    """
    def __init__(self, n_channels: int, n_classes: int, img_size: Tuple[int, int] = (224, 224),
                 depths: Tuple[int, ...] = (2, 2, 2, 2), num_heads: Tuple[int, ...] = (4, 8, 16, 32),
                 window_size: Tuple[int, int] = (7, 7), mlp_ratio: float = 4.0, qkv_bias: bool = True,
                 drop_rate: float = 0.0):
        super(SwinUNETR, self).__init__(n_channels, n_classes)
        self.img_size = img_size
        self.num_layers = len(depths)
        self.embed_dim = 96
        self.patch_size = (4, 4)
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=self.patch_size,
                                     in_chans=n_channels, embed_dim=self.embed_dim)
        
        patches_resolution = [img_size[0] // self.patch_size[0], img_size[1] // self.patch_size[1]]
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        
        # Transformer encoder
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate
            )
            self.layers.append(layer)
        
        # Decoder
        self.decoder0 = nn.Sequential(
            nn.Conv2d(self.embed_dim, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.decoder1 = nn.Sequential(
            nn.Conv2d(self.embed_dim * 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.decoder2 = nn.Sequential(
            nn.Conv2d(self.embed_dim * 4, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        self.decoder3 = nn.Sequential(
            nn.Conv2d(self.embed_dim * 8, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(self.embed_dim * 16, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        self.final = nn.Sequential(
            nn.Conv2d(32, n_classes, kernel_size=1)
        )
        
        self.norm = nn.LayerNorm(self.embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B = x.shape[0]
        x_size = (x.shape[2], x.shape[3])
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.view(B, x_size[0] // self.patch_size[0], x_size[1] // self.patch_size[1], self.embed_dim)
        
        # Transformer encoder
        x0 = x
        x1 = self.layers[0](x0)
        x2 = self.layers[1](x1)
        x3 = self.layers[2](x2)
        x4 = self.layers[3](x3)
        
        # Reshape for decoder
        x0 = x0.permute(0, 3, 1, 2).contiguous()
        x1 = x1.permute(0, 3, 1, 2).contiguous()
        x2 = x2.permute(0, 3, 1, 2).contiguous()
        x3 = x3.permute(0, 3, 1, 2).contiguous()
        x4 = x4.permute(0, 3, 1, 2).contiguous()
        
        # Decoder
        x0 = self.decoder0(x0)
        x1 = self.decoder1(x1)
        x2 = self.decoder2(x2)
        x3 = self.decoder3(x3)
        x4 = self.decoder4(x4)
        
        # Upsampling
        x = self.up4(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.up2(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x)
        x = torch.cat([x, x0], dim=1)
        
        # Final output
        x = self.final(x)
        return x


class NnUNet(BaseSegmentationModel):
    """
    nnU-Net адаптивная архитектура, соответствующая весам в файле NnUNet_best.pth
    """
    def __init__(self, n_channels: int, n_classes: int, deep_supervision: bool = False):
        super(NnUNet, self).__init__(n_channels, n_classes)
        self.deep_supervision = deep_supervision
        
        # Входной блок (inc) - структура соответствует весам
        self.inc = nn.Sequential()
        self.inc.add_module('conv', nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Identity(),  # Заполнитель для индекса 2
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ))
        
        # Блоки понижения размерности (down) - создаем структуру, соответствующую весам
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2)
        )
        self.down1.add_module('1', nn.Sequential())
        self.down1[1].add_module('conv', nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Identity(),  # Заполнитель для индекса 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ))
        
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2)
        )
        self.down2.add_module('1', nn.Sequential())
        self.down2[1].add_module('conv', nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Identity(),  # Заполнитель для индекса 2
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ))
        
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2)
        )
        self.down3.add_module('1', nn.Sequential())
        self.down3[1].add_module('conv', nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Identity(),  # Заполнитель для индекса 2
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ))
        
        # Центральный блок (bridge) - создаем структуру, соответствующую весам
        self.bridge = nn.Sequential(
            nn.MaxPool2d(2)
        )
        self.bridge.add_module('1', nn.Sequential())
        self.bridge[1].add_module('conv', nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Identity(),  # Заполнитель для индекса 2
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ))
        
        # Блоки повышения размерности (conv) - создаем структуру, соответствующую весам
        # В этой архитектуре нет отдельных up слоев, вместо этого используется интерполяция
        self.conv1 = nn.Sequential()
        self.conv1.add_module('conv', nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Identity(),  # Заполнитель для индекса 2
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ))
        
        self.conv2 = nn.Sequential()
        self.conv2.add_module('conv', nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Identity(),  # Заполнитель для индекса 2
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ))
        
        self.conv3 = nn.Sequential()
        self.conv3.add_module('conv', nn.Sequential(
            nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Identity(),  # Заполнитель для индекса 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ))
        
        self.conv4 = nn.Sequential()
        self.conv4.add_module('conv', nn.Sequential(
            nn.Conv2d(64 + 32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Identity(),  # Заполнитель для индекса 2
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        ))
        
        # Выходной слой
        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)
        
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Инициализация весов
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Входной блок
        x0 = self.inc(x)
        
        # Блоки понижения размерности
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        
        # Центральный блок
        x4 = self.bridge(x3)
        
        # Блоки повышения размерности с использованием интерполяции
        x = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x3, x], dim=1)
        x = self.conv1(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x2, x], dim=1)
        x = self.conv2(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x1, x], dim=1)
        x = self.conv3(x)
        
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x0, x], dim=1)
        x = self.conv4(x)
        
        # Выходной слой
        logits = self.outc(x)
        return logits


class UNetPlusPlus(BaseSegmentationModel):
    """
    UNet++ архитектура с вложенными плотными соединениями
    """
    def __init__(self, n_channels: int, n_classes: int, bilinear: bool = False, deep_supervision: bool = False):
        super(UNetPlusPlus, self).__init__(n_channels, n_classes)
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision
        self.nb_filter = [64, 128, 256, 512, 1024]
        
        # Encoder
        self.conv0_0 = DoubleConv(n_channels, self.nb_filter[0])
        self.conv1_0 = DoubleConv(self.nb_filter[0], self.nb_filter[1])
        self.conv2_0 = DoubleConv(self.nb_filter[1], self.nb_filter[2])
        self.conv3_0 = DoubleConv(self.nb_filter[2], self.nb_filter[3])
        self.conv4_0 = DoubleConv(self.nb_filter[3], self.nb_filter[4])
        
        # Плотные соединения в декодере
        self.conv0_1 = DoubleConv(self.nb_filter[0] + self.nb_filter[1], self.nb_filter[0])
        self.conv1_1 = DoubleConv(self.nb_filter[1] + self.nb_filter[2], self.nb_filter[1])
        self.conv2_1 = DoubleConv(self.nb_filter[2] + self.nb_filter[3], self.nb_filter[2])
        self.conv3_1 = DoubleConv(self.nb_filter[3] + self.nb_filter[4], self.nb_filter[3])
        
        self.conv0_2 = DoubleConv(self.nb_filter[0] * 2 + self.nb_filter[1], self.nb_filter[0])
        self.conv1_2 = DoubleConv(self.nb_filter[1] * 2 + self.nb_filter[2], self.nb_filter[1])
        self.conv2_2 = DoubleConv(self.nb_filter[2] * 2 + self.nb_filter[3], self.nb_filter[2])
        
        self.conv0_3 = DoubleConv(self.nb_filter[0] * 3 + self.nb_filter[1], self.nb_filter[0])
        self.conv1_3 = DoubleConv(self.nb_filter[1] * 3 + self.nb_filter[2], self.nb_filter[1])
        
        self.conv0_4 = DoubleConv(self.nb_filter[0] * 4 + self.nb_filter[1], self.nb_filter[0])
        
        # Final layers
        if deep_supervision:
            self.final1 = OutConv(self.nb_filter[0], n_classes)
            self.final2 = OutConv(self.nb_filter[0], n_classes)
            self.final3 = OutConv(self.nb_filter[0], n_classes)
            self.final4 = OutConv(self.nb_filter[0], n_classes)
        self.final = OutConv(self.nb_filter[0], n_classes)
        
        # Pooling и upsampling
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], dim=1))
        
        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], dim=1))
        
        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], dim=1))
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], dim=1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], dim=1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], dim=1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], dim=1))
        
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        
        return self.final(x0_4)


class EfficientNetUNet(BaseSegmentationModel):
    """
    EfficientNet-UNet архитектура с предобученным энкодером
    """
    def __init__(self, n_channels: int, n_classes: int, encoder_name: str = 'efficientnet-b0',
                 pretrained: bool = True):
        super(EfficientNetUNet, self).__init__(n_channels, n_classes)
        
        try:
            import efficientnet_pytorch
            self.encoder = efficientnet_pytorch.EfficientNet.from_pretrained(encoder_name) if pretrained else efficientnet_pytorch.EfficientNet.from_name(encoder_name)
        except ImportError:
            raise ImportError("efficientnet_pytorch is required. Install with: pip install efficientnet_pytorch")
        
        # Изменение первого слоя для поддержки n_channels
        if n_channels != 3:
            self.encoder._conv_stem = nn.Conv2d(
                n_channels, 32, kernel_size=3, stride=2, bias=False, padding=1
            )
        
        # Получение выходных каналов энкодера
        self.encoder_channels = [
            32,   # stem
            16,   # blocks1
            24,   # blocks2
            40,   # blocks3
            112,  # blocks4
            320   # blocks5
        ]
        
        # Декодер
        self.decoder_channels = [256, 128, 64, 32, 16]
        
        # Декодерные блоки
        self.up5 = nn.ConvTranspose2d(self.encoder_channels[-1], self.decoder_channels[0], kernel_size=2, stride=2)
        self.dec5 = DoubleConv(self.encoder_channels[-2] + self.decoder_channels[0], self.decoder_channels[0])
        
        self.up4 = nn.ConvTranspose2d(self.decoder_channels[0], self.decoder_channels[1], kernel_size=2, stride=2)
        self.dec4 = DoubleConv(self.encoder_channels[-3] + self.decoder_channels[1], self.decoder_channels[1])
        
        self.up3 = nn.ConvTranspose2d(self.decoder_channels[1], self.decoder_channels[2], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(self.encoder_channels[-4] + self.decoder_channels[2], self.decoder_channels[2])
        
        self.up2 = nn.ConvTranspose2d(self.decoder_channels[2], self.decoder_channels[3], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(self.encoder_channels[-5] + self.decoder_channels[3], self.decoder_channels[3])
        
        self.up1 = nn.ConvTranspose2d(self.decoder_channels[3], self.decoder_channels[4], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(self.encoder_channels[-6] + self.decoder_channels[4], self.decoder_channels[4])
        
        # Финальный слой
        self.final = nn.Conv2d(self.decoder_channels[4], n_classes, kernel_size=1)
        
    def forward(self, x):
        # Энкодер
        features = []
        
        # Stem
        x = self.encoder._conv_stem(x)
        x = self.encoder._bn0(x)
        x = self.encoder._swish(x)
        features.append(x)
        
        # Blocks
        for idx, block in enumerate(self.encoder._blocks):
            x = block(x)
            if idx in [2, 4, 10, 15]:  # Индексы для получения признаков разных уровней
                features.append(x)
        
        # Декодер
        x5 = features[-1]  # 320
        x4 = features[-2]  # 112
        x3 = features[-3]  # 40
        x2 = features[-4]  # 24
        x1 = features[-5]  # 16
        
        # Декодирование
        d5 = self.up5(x5)
        d5 = torch.cat([d5, x4], dim=1)
        d5 = self.dec5(d5)
        
        d4 = self.up4(d5)
        d4 = torch.cat([d4, x3], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, features[0]], dim=1)
        d1 = self.dec1(d1)
        
        # Финальный выход
        return self.final(d1)


class TransformerBlock(nn.Module):
    """
    Базовый Transformer блок для TransUNet и CoTr
    """
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        # MLP
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        
        return x


class TransUNet(BaseSegmentationModel):
    """
    TransUNet архитектура (гибрид CNN-Transformer)
    """
    def __init__(self, n_channels: int, n_classes: int, img_size: Tuple[int, int] = (224, 224),
                 patch_size: int = 16, num_layers: int = 4, num_heads: int = 8,
                 embed_dim: int = 768, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(TransUNet, self).__init__(n_channels, n_classes)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # CNN энкодер для извлечения локальных признаков
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            256, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Positional encoding
        h_patches = img_size[0] // (2 * 2 * 2 * patch_size)  # Учитываем пулинги в CNN
        w_patches = img_size[1] // (2 * 2 * 2 * patch_size)
        num_patches = h_patches * w_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer encoder
        self.transformer_encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Декодер
        self.decoder_channels = [256, 128, 64, 32]
        
        self.up3 = nn.ConvTranspose2d(embed_dim, self.decoder_channels[0], kernel_size=2, stride=2)
        self.dec3 = DoubleConv(embed_dim + self.decoder_channels[0], self.decoder_channels[0])
        
        self.up2 = nn.ConvTranspose2d(self.decoder_channels[0], self.decoder_channels[1], kernel_size=2, stride=2)
        self.dec2 = DoubleConv(self.decoder_channels[0] + self.decoder_channels[1], self.decoder_channels[1])
        
        self.up1 = nn.ConvTranspose2d(self.decoder_channels[1], self.decoder_channels[2], kernel_size=2, stride=2)
        self.dec1 = DoubleConv(self.decoder_channels[1] + self.decoder_channels[2], self.decoder_channels[2])
        
        self.up0 = nn.ConvTranspose2d(self.decoder_channels[2], self.decoder_channels[3], kernel_size=2, stride=2)
        self.dec0 = DoubleConv(self.decoder_channels[2] + self.decoder_channels[3], self.decoder_channels[3])
        
        # Финальный слой
        self.final = nn.Conv2d(self.decoder_channels[3], n_classes, kernel_size=1)
        
        # Инициализация
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # CNN энкодер
        x1 = self.cnn_encoder[0:4](x)  # После первого пулинга
        x2 = self.cnn_encoder[4:10](x1)  # После второго пулинга
        x3 = self.cnn_encoder[10:](x2)  # После третьего пулинга
        
        # Patch embedding
        x = self.patch_embed(x3)  # [B, embed_dim, h_patches, w_patches]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Добавление позиционного кодирования
        x = x + self.pos_embed
        
        # Transformer encoder
        for transformer_block in self.transformer_encoder:
            x = transformer_block(x)
        
        # Восстановление размерности
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Декодер
        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec3(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec2(x)
        
        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec1(x)
        
        x = self.up0(x)
        x = torch.cat([x, x], dim=1)  # Здесь можно добавить skip connection с входом
        x = self.dec0(x)
        
        # Финальный выход
        return self.final(x)


class CoTr(BaseSegmentationModel):
    """
    CoTr (Convolutional Transformer) архитектура
    """
    def __init__(self, n_channels: int, n_classes: int, img_size: Tuple[int, int] = (224, 224),
                 patch_size: int = 16, num_layers: int = 4, num_heads: int = 8,
                 embed_dim: int = 768, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super(CoTr, self).__init__(n_channels, n_classes)
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # CNN энкодер для извлечения признаков разных уровней
        self.enc1 = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        
        # Patch embedding для самого глубокого уровня
        self.patch_embed = nn.Conv2d(
            512, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Positional encoding
        h_patches = img_size[0] // (2 * 2 * 2 * 2 * patch_size)  # Учитываем все пулинги
        w_patches = img_size[1] // (2 * 2 * 2 * 2 * patch_size)
        num_patches = h_patches * w_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # Transformer encoder
        self.transformer_encoder = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Декодер с attention механизмами
        self.up3 = nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(embed_dim + 256, 256)
        self.att3 = AttentionBlock(256, 256, 128)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256 + 128, 128)
        self.att2 = AttentionBlock(128, 128, 64)
        
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128 + 64, 64)
        self.att1 = AttentionBlock(64, 64, 32)
        
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = DoubleConv(64 + 32, 32)
        
        # Финальный слой
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)
        
        # Инициализация
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # CNN энкодер
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        
        # Patch embedding
        x = self.patch_embed(x4)  # [B, embed_dim, h_patches, w_patches]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Добавление позиционного кодирования
        x = x + self.pos_embed
        
        # Transformer encoder
        for transformer_block in self.transformer_encoder:
            x = transformer_block(x)
        
        # Восстановление размерности
        x = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Декодер с attention
        x = self.up3(x)
        x = torch.cat([x, x3], dim=1)
        x3 = self.dec3(x)
        x3 = self.att3(x3, x3)
        
        x = self.up2(x3)
        x = torch.cat([x, x2], dim=1)
        x2 = self.dec2(x)
        x2 = self.att2(x2, x2)
        
        x = self.up1(x2)
        x = torch.cat([x, x1], dim=1)
        x1 = self.dec1(x)
        x1 = self.att1(x1, x1)
        
        x = self.up0(x1)
        x = torch.cat([x, x], dim=1)  # Здесь можно добавить skip connection с входом
        x = self.dec0(x)
        
        # Финальный выход
        return self.final(x)


def get_model(model_name: str, **kwargs) -> nn.Module:
    """
    Функция для получения модели по имени
    
    Args:
        model_name: Имя модели ('unet', 'attention_unet', 'unet3d', 'resunet', 'swin_unetr', 'nnunet',
                              'unet_plus_plus', 'efficientnet_unet', 'transunet', 'cotr')
        **kwargs: Дополнительные параметры для модели
        
    Returns:
        nn.Module: Запрошенная модель
    """
    if model_name.lower() == 'unet':
        return UNet(**kwargs)
    elif model_name.lower() == 'attention_unet':
        return AttentionUNet(**kwargs)
    elif model_name.lower() == 'unet3d':
        return UNet3D(**kwargs)
    elif model_name.lower() == 'resunet':
        return ResUNet(**kwargs)
    elif model_name.lower() == 'swin_unetr':
        return SwinUNETR(**kwargs)
    elif model_name.lower() == 'nnunet':
        return NnUNet(**kwargs)
    elif model_name.lower() == 'unet_plus_plus':
        return UNetPlusPlus(**kwargs)
    elif model_name.lower() == 'efficientnet_unet':
        return EfficientNetUNet(**kwargs)
    elif model_name.lower() == 'transunet':
        return TransUNet(**kwargs)
    elif model_name.lower() == 'cotr':
        return CoTr(**kwargs)
    else:
        raise ValueError(f"Unknown model name: {model_name}. Available models: 'unet', 'attention_unet', 'unet3d', 'resunet', 'swin_unetr', 'nnunet', 'unet_plus_plus', 'efficientnet_unet', 'transunet', 'cotr'")


if __name__ == "__main__":
    # Тестирование моделей
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2D модели
    x = torch.randn(1, 1, 224, 224).to(device)
    
    models_2d = {
        'UNet': UNet(n_channels=1, n_classes=1),
        'AttentionUNet': AttentionUNet(n_channels=1, n_classes=1),
        'ResUNet': ResUNet(n_channels=1, n_classes=1),
        'SwinUNETR': SwinUNETR(n_channels=1, n_classes=1, img_size=(224, 224)),
        'NnUNet': NnUNet(n_channels=1, n_classes=1),
        'UNetPlusPlus': UNetPlusPlus(n_channels=1, n_classes=1),
        'TransUNet': TransUNet(n_channels=1, n_classes=1, img_size=(224, 224)),
        'CoTr': CoTr(n_channels=1, n_classes=1, img_size=(224, 224))
    }
    
    # EfficientNet-UNet требует отдельной обработки из-за возможных ошибок импорта
    try:
        models_2d['EfficientNetUNet'] = EfficientNetUNet(n_channels=1, n_classes=1)
    except ImportError as e:
        print(f"EfficientNetUNet skipped: {e}")
    
    for name, model in models_2d.items():
        model = model.to(device)
        with torch.no_grad():
            y = model(x)
        print(f"{name}: input {x.shape} -> output {y.shape}")
        
        # Вывод информации о модели
        model_info = model.get_model_info()
        print(f"  Parameters: {model_info['total_parameters']:,} total, {model_info['trainable_parameters']:,} trainable")
        print(f"  Model size: {model_info['model_size_mb']:.2f} MB")
    
    # 3D модель
    x3d = torch.randn(1, 1, 32, 128, 128).to(device)
    model_3d = UNet3D(n_channels=1, n_classes=1).to(device)
    with torch.no_grad():
        y3d = model_3d(x3d)
    print(f"UNet3D: input {x3d.shape} -> output {y3d.shape}")
    
    # Тестирование функции get_model
    print("\nTesting get_model function:")
    for model_name in ['unet', 'attention_unet', 'unet_plus_plus', 'transunet']:
        try:
            model = get_model(model_name, n_channels=1, n_classes=1)
            model = model.to(device)
            with torch.no_grad():
                y = model(x)
            print(f"{model_name}: input {x.shape} -> output {y.shape}")
        except Exception as e:
            print(f"Error with {model_name}: {e}")
"""model.py — ResNet U-Net style colorization model (supports ResNet-18 and ResNet-34)"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, dropout=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.conv = nn.Sequential(
            ConvBlock(out_channels + skip_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.dropout = nn.Dropout2d(0.2) if dropout else nn.Identity()

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.dropout(x)
        return self.conv(x)


class ResNetColorizer(nn.Module):
    def __init__(self, pretrained=True, backbone='resnet34'):
        super().__init__()

        if backbone == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet18(weights=weights)
        else:
            weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = resnet34(weights=weights)

        # Encoder: keep all initial stages
        self.conv1 = backbone.conv1
        self.bn1   = backbone.bn1
        self.relu  = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Dilated bottleneck (from layer4 output, 512 channels, 8x8)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        # Decoder (U-Net style with skip connections)
        self.decoder1 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=256, dropout=True)
        self.decoder2 = DecoderBlock(in_channels=256, skip_channels=128, out_channels=128, dropout=True)
        self.decoder3 = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.decoder4 = DecoderBlock(in_channels=64, skip_channels=64, out_channels=64)
        self.decoder5 = DecoderBlock(in_channels=64, skip_channels=0, out_channels=32)

        # Final prediction to 2 ab channels (prediction in normalized [-1,+1] range via tanh)
        self.head = nn.Sequential(
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 2, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, l):
        # l: (B, 1, H, W) in normalized Lab L range
        x = l.repeat(1, 3, 1, 1)  # make 3-channel grayscale for ResNet

        x0 = self.relu(self.bn1(self.conv1(x)))  # (B, 64, H/2, W/2)
        x1 = self.maxpool(x0)                    # (B, 64, H/4, W/4)
        x1 = self.layer1(x1)                     # (B, 64, H/4, W/4)
        x2 = self.layer2(x1)                     # (B, 128, H/8, W/8)
        x3 = self.layer3(x2)                     # (B, 256, H/16, W/16)
        x4 = self.layer4(x3)                     # (B, 512, H/32, W/32)

        x = self.bottleneck(x4)

        x = self.decoder1(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)
        x = self.decoder4(x, x0)
        x = self.decoder5(x, None)

        return self.head(x)

    def freeze_encoder(self):
        for param in [*self.conv1.parameters(), *self.bn1.parameters(), *self.layer1.parameters(), *self.layer2.parameters(), *self.layer3.parameters(), *self.layer4.parameters()]:
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in [*self.conv1.parameters(), *self.bn1.parameters(), *self.layer1.parameters(), *self.layer2.parameters(), *self.layer3.parameters(), *self.layer4.parameters()]:
            param.requires_grad = True


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = ResNetColorizer(pretrained=True)
    x = torch.randn(1, 1, 256, 256)
    y = model(x)
    print("output shape:", y.shape)
    print("trainable params:", count_parameters(model))

import torch
import torch.nn.functional as F
from torch import nn

from ..utils import initialize_weights


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        self.encode = nn.Sequential(*layers)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        enc = self.encode(x)
        x = self.pool(enc)
        return enc, x


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels):
        super(_DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, middle_channels, kernel_size=2, stride=2)

        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, top, bottom):
        bottom = self.upconv(bottom)
        return self.decode(torch.cat([top, bottom], dim=1))


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(3, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.dec4 = _DecoderBlock(1024, 512)
        self.dec3 = _DecoderBlock(512, 256)
        self.dec2 = _DecoderBlock(256, 128)
        self.dec1 = _DecoderBlock(128, 64)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        initialize_weights(self)

    def forward(self, x):
        enc1, x1 = self.enc1(x)
        enc2, x2 = self.enc2(x1)
        enc3, x3 = self.enc3(x2)
        enc4, x4 = self.enc4(x3)
        center = self.center(x4)
        dec4 = self.dec4(top=enc4, bottom=center)
        dec3 = self.dec3(top=enc3, bottom=dec4)
        dec2 = self.dec2(top=enc2, bottom=dec3)
        dec1 = self.dec1(top=enc1, bottom=dec2)
        final = self.final(dec1)
        return final

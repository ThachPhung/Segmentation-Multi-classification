import torch
import torch.nn as nn
import torch.nn.functional as F


"""Function Blocks"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv_block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            ConvBlock(in_channels, out_channels, kernel_size, padding),
        )
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, bilinear=True):
        super().__init__()
        if bilinear:
            self.conv_trans = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, padding)
        else:
            self.conv_trans = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv_block = ConvBlock(in_channels, out_channels, kernel_size, padding)

    def forward(self, x1, x2):
        # B, C, H, W
        x1= self.conv_trans(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1) # concat by channel
        return self.conv_block(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)

"""Model Architecture"""
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super().__init__()
        self.inc = ConvBlock(n_channels, 64, kernel_size=3, padding=1)
        self.down1 = Encoder(64, 128, 3, 1)
        self.down2 = Encoder(128, 256, 3, 1)
        self.down3 = Encoder(256, 512, 3, 1)
        factor = 2 if bilinear else 1
        self.down4 = Encoder(512, 1024 // factor, 3, 1)
        self.bottom_neck = ConvBlock(1024//factor, 1024//factor, kernel_size=1, padding=0)
        self.up1 = Decoder(1024, 512//factor, 3, 1, bilinear=bilinear)
        self.up2 = Decoder(512, 256//factor, 3, 1, bilinear=bilinear)
        self.up3 = Decoder(256, 128//factor, 3, 1, bilinear=bilinear)
        self.up4 = Decoder(128, 64, 3, 1, bilinear=bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x) # 64
        x2 = self.down1(x1) # 128
        x3 = self.down2(x2) # 256
        x4 = self.down3(x3) #512
        x5 = self.down4(x4) # 1024
        x5 = self.bottom_neck(x5) # 1024
        x = self.up1(x5, x4) # 512
        x = self.up2(x, x3) # 256
        x = self.up3(x, x2) # 128
        logits = self.up4(x, x1) # 64
        return self.outc(logits)

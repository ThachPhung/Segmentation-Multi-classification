import torch
import torch.nn as nn


class SE_BLOCK(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()

        y = self.avg_pool(x).view(b, c)

        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


# ================================
# Global Attention Mechanism
# ================================
class ChannelAttention(nn.Module):
    def __init__(self, channels, r: int = 16):
        super().__init__()
        self.channels = channels
        self.hidden_channels = max(channels // r, 1)

        self.ffn1 = nn.Linear(self.channels, self.hidden_channels)
        self.act = nn.SiLU()

        self.ffn2 = nn.Linear(self.hidden_channels, self.channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = x.shape[2]
        w = x.shape[3]

        # Change from chw -> whc
        x_perm = x.permute(0, 3, 2, 1)
        x_perm = x_perm.reshape(-1, w * h, self.channels)  # (batch, wh, channel)

        # MLP
        attn_w = self.ffn1(x_perm)  # (batch, wh, hidden channel)
        attn_w = self.act(attn_w)
        attn_w = self.ffn2(attn_w)

        # Convert back to chw
        attn_w = attn_w.reshape(-1, w, h, self.channels)
        attn_w = attn_w.permute(0, 3, 2, 1)
        attn_w = self.sigmoid(attn_w)  # (batch, channel, w, h)
        return (attn_w * x) + (0.01 * x)


class SpatialAttention(nn.Module):
    def __init__(self, channels, pool_kernel=7, r: int = 16):
        super().__init__()
        self.channels = channels
        self.hidden_channels = max(channels // r, 1)

        self.conv1 = nn.Conv2d(self.channels, self.hidden_channels, kernel_size=pool_kernel, padding=pool_kernel // 2)
        self.bn = nn.BatchNorm2d(self.hidden_channels)
        self.act = nn.SiLU()

        self.conv2 = nn.Conv2d(self.hidden_channels, self.channels, kernel_size=pool_kernel, padding=pool_kernel // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn_weight = self.conv1(x)
        attn_weight = self.act(self.bn(attn_weight))

        attn_weight = self.sigmoid(self.conv2(attn_weight))
        return (attn_weight * x) + (0.01 * x)


class GlobalAttentionMechanism(nn.Module):
    def __init__(self, c1, c2=None, pool_kernel=7, r: int = 16):
        if c2 is None:
            c2 = c1
        assert c1 == c2

        super().__init__()
        self.channel_attn = ChannelAttention(c1, r=r)
        self.spatial_attn = SpatialAttention(c1, pool_kernel=pool_kernel, r=r)

    def forward(self, x):
        x = self.channel_attn(x)
        return self.spatial_attn(x)

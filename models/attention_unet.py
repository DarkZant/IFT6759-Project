import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class AttentionBlock(nn.Module):
    """
    Attention Gate: Filters the skip connection (x) based on the gating signal (g)
    from the layer below, suppressing background noise.
    """

    def __init__(self, F_g, F_l, F_int):
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

        # In case dimensions don't perfectly match due to pooling/upsampling math
        if g1.shape != x1.shape:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=16, out_channels=3, features=[16, 32, 64, 128]):
        super(AttentionUNet, self).__init__()
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder (Downsampling)
        self.Conv1 = ConvBlock(in_channels, features[0])
        self.Conv2 = ConvBlock(features[0], features[1])
        self.Conv3 = ConvBlock(features[1], features[2])
        self.Conv4 = ConvBlock(features[2], features[3])

        # Bottleneck
        self.Up5 = nn.ConvTranspose2d(features[3], features[2], kernel_size=2, stride=2)
        self.Att5 = AttentionBlock(F_g=features[2], F_l=features[2], F_int=features[1])
        self.Up_conv5 = ConvBlock(features[3], features[2])

        self.Up4 = nn.ConvTranspose2d(features[2], features[1], kernel_size=2, stride=2)
        self.Att4 = AttentionBlock(F_g=features[1], F_l=features[1], F_int=features[0])
        self.Up_conv4 = ConvBlock(features[2], features[1])

        self.Up3 = nn.ConvTranspose2d(features[1], features[0], kernel_size=2, stride=2)
        self.Att3 = AttentionBlock(F_g=features[0], F_l=features[0], F_int=features[0] // 2)
        self.Up_conv3 = ConvBlock(features[1], features[0])

        self.Conv_1x1 = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        e1 = self.Conv1(x)

        e2 = self.Maxpool(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool(e3)
        e4 = self.Conv4(e4)

        # Decoder with Attention Gates
        d5 = self.Up5(e4)

        # Ensure sizes match before attention
        if d5.shape != e3.shape:
            d5 = F.interpolate(d5, size=e3.shape[2:], mode='bilinear', align_corners=True)

        x5 = self.Att5(g=d5, x=e3)
        d5 = torch.cat((x5, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        if d4.shape != e2.shape:
            d4 = F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=True)

        x4 = self.Att4(g=d4, x=e2)
        d4 = torch.cat((x4, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        if d3.shape != e1.shape:
            d3 = F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=True)

        x3 = self.Att3(g=d3, x=e1)
        d3 = torch.cat((x3, d3), dim=1)
        d3 = self.Up_conv3(d3)

        out = self.Conv_1x1(d3)
        return out
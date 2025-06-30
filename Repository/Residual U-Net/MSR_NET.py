import torch.nn.functional as F
import torch.nn as nn
p = 0.2
class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class MSRNet(nn.Module):
    def __init__(self, in_=1, feature=32, p=3, out_channels=2):
        super(MSRNet, self).__init__()
        self.inchannel=in_
        self.down_conv1 = self._conv(in_, feature, stride=1)
        self.down_conv2 = self._conv(feature, feature*2, stride=1)
        self.down_conv3 = self._conv(feature*2, feature * 3, stride=1)

        self.layer1 = self.make_layer(ResidualBlock, feature,  p-2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, feature*2, p-1, stride=1)
        self.layer3 = self.make_layer(ResidualBlock, feature*3, p, stride=1)

        self.up_conv3 = nn.Sequential(ResidualBlock(feature * 3, feature * 2, 1))
        self.up_conv2 = nn.Sequential(ResidualBlock(feature*2, feature, 1))
        self.up_conv1 = nn.Sequential(ResidualBlock(feature, out_channels, 1))

        self.Upsample = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)
        x2 = F.max_pool2d(x2, 2)
        x3 = self.down_conv3(x2)
        x3 = F.max_pool2d(x3, 2)

        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        x3 = self.layer3(x3)

        x3_up = self.up_conv3(self.Upsample(x3))
        x2_up = self.up_conv2(x3_up+x2)
        out = self.up_conv1(self.Upsample(x2_up) + x1)

        return out
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        self.inchannel = channels
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def _conv(self, in_channel, out_channel, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )

class MSRNet_TL(nn.Module):
    """
    MSRNet adapted for TLoss:
    - Outputs a single-channel residual map (B, H, W)
    - No final activation (raw logits)
    """
    def __init__(self, in_channels=1, feature=32, p=3):
        super(MSRNet_TL, self).__init__()
        # Downsampling path
        self.down_conv1 = self._conv(in_channels, feature)
        self.down_conv2 = self._conv(feature, feature * 2)
        self.down_conv3 = self._conv(feature * 2, feature * 3)

        # Residual layers
        self.layer1 = self.make_layer(ResidualBlock, feature, p - 2)
        self.layer2 = self.make_layer(ResidualBlock, feature * 2, p - 1)
        self.layer3 = self.make_layer(ResidualBlock, feature * 3, p)

        # Upsampling path
        self.up_block3 = ResidualBlock(feature * 3, feature * 2)
        self.up_block2 = ResidualBlock(feature * 2, feature)

        # Final head: single-channel output, no activation
        self.head = nn.Conv2d(feature, 1, kernel_size=3, padding=1, bias=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        # Encoder
        x1 = self.down_conv1(x)              # (B, f, H, W)
        x2 = self.down_conv2(x1)             # (B, 2f, H, W)
        x2 = F.max_pool2d(x2, 2)             # (B, 2f, H/2, W/2)
        x3 = self.down_conv3(x2)             # (B, 3f, H/2, W/2)
        x3 = F.max_pool2d(x3, 2)             # (B, 3f, H/4, W/4)

        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        x3 = self.layer3(x3)

        # Decoder
        x3_up = self.up_block3(self.upsample(x3))       # (B, 2f, H/2, W/2)
        x2_up = self.up_block2(x3_up + x2)               # (B, f, H/2, W/2)
        out = self.head(self.upsample(x2_up) + x1)       # (B, 1, H, W)

        return out.squeeze(1)  # (B, H, W)

    def make_layer(self, block, channels, num_blocks, stride=1):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(channels, channels, stride))
            stride = 1
        return nn.Sequential(*layers)

    def _conv(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

class MSRNet_2_classes(nn.Module):
    def __init__(self, in_=1, feature=32, p=3, out_channels=3):
        super(MSRNet_2_classes, self).__init__()
        self.inchannel=in_
        self.down_conv1 = self._conv(in_, feature, stride=1)
        self.down_conv2 = self._conv(feature, feature*2, stride=1)
        self.down_conv3 = self._conv(feature*2, feature * 3, stride=1)

        self.layer1 = self.make_layer(ResidualBlock, feature,  p-2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, feature*2, p-1, stride=1)
        self.layer3 = self.make_layer(ResidualBlock, feature*3, p, stride=1)

        self.up_conv3 = nn.Sequential(ResidualBlock(feature * 3, feature * 2, 1))
        self.up_conv2 = nn.Sequential(ResidualBlock(feature*2, feature, 1))
        self.up_conv1 = nn.Sequential(ResidualBlock(feature, out_channels, 1))

        self.Upsample = nn.Upsample(scale_factor=2, mode='nearest')
    def forward(self, x):
        x1 = self.down_conv1(x)
        x2 = self.down_conv2(x1)
        x2 = F.max_pool2d(x2, 2)
        x3 = self.down_conv3(x2)
        x3 = F.max_pool2d(x3, 2)

        x1 = self.layer1(x1)
        x2 = self.layer2(x2)
        x3 = self.layer3(x3)

        x3_up = self.up_conv3(self.Upsample(x3))
        x2_up = self.up_conv2(x3_up+x2)
        out = self.up_conv1(self.Upsample(x2_up) + x1)

        return out
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        self.inchannel = channels
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def _conv(self, in_channel, out_channel, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )



def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
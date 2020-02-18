import math
import torch
from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, scale_factor, nums_residuals=20):
        upsample_block_num = int(math.log(scale_factor, 2))
        super(Generator, self).__init__()

        # 先验网络
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        resnet_blocks1 = []
        for _ in range(nums_residuals):
            resnet_blocks1.append(ResidualBlock(64))
        self.block2 = nn.Sequential(*resnet_blocks1)

        # self.block3 = nn.Sequential(
        #     nn.ReLU(),
        #     nn.BatchNorm2d(64),
        #     nn.Conv2d(64, 3, kernel_size=3, padding=1),

        # )
        self.block3 = nn.Sequential(
            UpsampleBLock(64, 4),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)
        )


        self.block4 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=2),
            # nn.BatchNorm2d(64),
            nn.ReLU()
        )

        resnet_blocks2 = []
        for _ in range(3):
            resnet_blocks2.append(ResidualBlock(64))
        self.block5 = nn.Sequential(*resnet_blocks2)

        # hourglass layer
        self.hg1 = Hourglass(Bottleneck, 3, 64//2, 4)
        self.hg2 = Hourglass(Bottleneck, 3, 64//2, 4)

        resnet_blocks3 = []
        for _ in range(3):
            resnet_blocks3.append(ResidualBlock(64))
        self.block6 = nn.Sequential(*resnet_blocks3)


        block7 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block7.append(nn.Conv2d(64, 3, kernel_size=3, padding=1, stride=2))
        self.block7 = nn.Sequential(*block7)




    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        # print(x.size())
        x = self.block3(x)
        # print(x.size())
        # exit()
        mid_x = x
        x = self.block4(x)
        # print(x.size())
        x = self.block5(x)
        x = self.hg1(x)
        x = self.hg2(x)
        x = self.block6(x)
        x = self.block7(x)
        # print(mid_x.size(), x.size())
        # exit()
        # return midx, x # 不知道这样子改正行不行？？？？
        return mid_x, x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        # residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        # residual = self.bn2(residual)

        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        # self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # out = self.bn1(x)
        out = self.relu(x)
        out = self.conv1(out)

        # out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        # out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class Hourglass(nn.Module):
    # depth 代表第几层 4 128 4
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    # 4 128 4
    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth): # 0 1 2 3
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            # 最底层多经过了3个卷积层
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        # 走上面
        up1 = self.hg[n-1][0](x)
        # 缩小为原来的1/2
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            # 递归传播
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            # 最里层
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        # print("epoch ", n)
        # print(up1.size())
        # print(up2.size())
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)

if __name__ == "__main__":
    netG = Generator(4)
    # print('# generator parameters:', sum(param.numel() for param in netG.parameters()))
    netD = Discriminator()
    print(netG)
    print(netD)
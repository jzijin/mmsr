import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import arch_util as arch_util

# import arch_util as arch_util

class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.ReLU(inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5],
                                     0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = 0.2 * out + residual
        out = self.relu(out)
        return out


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

class IncNet_block(nn.Module):
    """
        5个google inception 方式
    """

    def __init__(self, nf=64, gc=32, n_bottle_neck=3, n_res_net=5, n_rrdb=2):
        super(IncNet_block, self).__init__()

        bottle_neck_block = functools.partial(Bottleneck, inplanes=nf, planes=nf//4)
        self.bottle_neck = arch_util.make_layer(bottle_neck_block, n_bottle_neck)

        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.res_net = arch_util.make_layer(basic_block, n_res_net)

        RRDB_block_f = functools.partial(ResidualDenseBlock_5C, nf=nf, gc=gc)
        self.RRDB_trunk = arch_util.make_layer(RRDB_block_f, n_rrdb)

        self.relu = nn.ReLU(inplace=True)
        self.conv_last = nn.Conv2d(nf*3, nf, kernel_size=1, bias=True)

    def forward(self, x):
        out_bottle = self.bottle_neck(x)
        out_res = self.res_net(x)
        out_rrdb = self.RRDB_trunk(x)
        out_inc = torch.cat([out_bottle, out_res, out_rrdb], dim=1)
        out_inc = self.relu(self.conv_last(out_inc))
        return out_inc*0.2 + x


class IncNet(nn.Module):
    """
        the nb is fix TODO: change the nb as parameters
    """
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=5, gc=32):
        super(IncNet, self).__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        inc_block_f1 = functools.partial(IncNet_block, nf=nf, gc=gc)
        self.inc_trunk1 = arch_util.make_layer(inc_block_f1, 2)

        inc_block_f2 = functools.partial(IncNet_block, nf=nf, gc=gc)
        self.inc_trunk2 = arch_util.make_layer(inc_block_f2, 2)

        self.inc_trunk3 = IncNet_block(nf=nf, gc=gc)
        self.upsample1 = UpsampleBLock(nf, 2)
        self.upsample2 = UpsampleBLock(nf, 2)
        # TODO:
        # self.upsample3 = UpsampleBLock(nf, 2)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

    def forward(self, x):
        x = self.conv_first(x)
        res = x
        x = self.inc_trunk1(x)
        x = self.inc_trunk2(x)
        x = self.inc_trunk3(x)
        x += res
        x = self.upsample1(x)
        x = self.upsample2(x)

        # x = self.upsample3(x)
        x = self.conv_last(x)
        return x

if __name__ == "__main__":
    # net = ResidualDenseBlock_5C()
    # net = Bottleneck(64)
    # basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=64)
    # net = arch_util.make_layer(basic_block, 5)
    net = IncNet()
    print(net)
    x = torch.randn(2,3,32, 32)
    y = net(x)
    print(y.size())

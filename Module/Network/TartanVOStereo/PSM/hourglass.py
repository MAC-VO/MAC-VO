# HourGlass: https://github.com/princeton-vl/pytorch_stacked_hourglass/blob/master/models/layers.py
from torch import nn


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, nf)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n - 1, nf, increase=0)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, nf)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(up1)
        low2 = self.low2(pool1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class Hourglass2(nn.Module):
    """
    Simplified Hourglass wo/ residule modules
    """

    def __init__(self, n, f, increase=0):
        super(Hourglass2, self).__init__()
        nf = f + increase
        self.up1 = Conv(f, nf)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass2(n - 1, nf, increase=0)
        else:
            self.low2 = Conv(nf, nf)
        self.low3 = Conv(nf, nf)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        up1 = self.up1(x)
        pool1 = self.pool1(up1)
        low2 = self.low2(pool1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2

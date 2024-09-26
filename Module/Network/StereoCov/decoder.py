import torch
import torch.nn as nn
import torch.nn.functional as F

from ..TartanVOStereo.PSM import Hourglass


class HourglassDecoder(nn.Module):

    def __init__(self, act_fun="relu", exp=False):
        super(HourglassDecoder, self).__init__()
        self.exp = exp
        if act_fun == "relu":
            self.actfun = F.relu
        elif act_fun == "selu":
            self.actfun = F.selu
        else:
            print("Unknown activate function", act_fun)
            self.actfun = F.relu
        self.deconv_c7 = nn.ConvTranspose2d(
            896, 320, kernel_size=4, stride=2, padding=1
        )  # 1/16
        self.deconv_c8 = nn.ConvTranspose2d(
            576, 192, kernel_size=4, stride=2, padding=1
        )  # 1/8
        self.conv_c8 = Hourglass(2, 192, 0)  # 1/8
        self.deconv_c9 = nn.ConvTranspose2d(
            384, 128, kernel_size=4, stride=2, padding=1
        )  # 1/4
        self.conv_c9 = Hourglass(2, 128, 0)  # 1/4
        self.deconv_c10 = nn.ConvTranspose2d(
            256, 64, kernel_size=4, stride=2, padding=1
        )  # 1/2
        self.conv_c10 = Hourglass(2, 64, 0)  # 1/2
        self.deconv_c11 = nn.ConvTranspose2d(
            128, 64, kernel_size=4, stride=2, padding=1
        )  # 1/1
        self.conv_c12 = nn.Conv2d(64, 16, kernel_size=1, padding=0)
        self.conv_c13 = nn.Conv2d(16, 1, kernel_size=1, padding=0)

        self.deconv_c7_2 = nn.ConvTranspose2d(
            512, 512, kernel_size=4, stride=2, padding=1
        )  # 1/32

    def forward(self, x, cats):
        cat0, cat1, cat2, cat3, cat4 = cats

        x = self.deconv_c7_2(x)  # 1/32 - 512
        x = self.actfun(x, inplace=True)
        x = torch.cat((x, cat4), dim=1)  # - 896
        x = self.deconv_c7(x)  # 1/16 - 320
        x = self.actfun(x, inplace=True)
        x = torch.cat((x, cat3), dim=1)  # - 576 28,40
        x = self.deconv_c8(x)  # 1/8 - 192
        x = self.actfun(x, inplace=True)
        x = self.conv_c8(x)
        x = torch.cat((x, cat2), dim=1)  # - 384 56,80
        x = self.deconv_c9(x)  # 1/4 - 128
        x = self.actfun(x, inplace=True)
        x = self.conv_c9(x)
        x = torch.cat((x, cat1), dim=1)  # - 256 112,160
        x = self.deconv_c10(x)  # 1/2 - 64
        x = self.actfun(x, inplace=True)
        x = self.conv_c10(x)
        x = torch.cat((x, cat0), dim=1)  # - 128 224,320
        x = self.deconv_c11(x)  # 1/1 - 64
        x = self.actfun(x, inplace=True)
        x = self.conv_c12(x)
        x = self.actfun(x, inplace=True)
        x = self.conv_c13(x)
        if self.exp:
            x = torch.exp(x)
        else:
            x = self.actfun(x, inplace=True)
        return x

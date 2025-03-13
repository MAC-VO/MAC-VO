import torch.nn as nn
from typing import List

    

class CNNCovHead(nn.Module):
    """
    Estimate the covariance using a CNN architecture.
    """
    def __init__(self, k_list: List[int] = [7, 7, 7], c_list: List[int] = [128, 64, 32, 2], 
                        s_list: List[int]|None = None, p_list: List[int]|None = None):
        super().__init__()
        assert len(k_list) == len(c_list) - 1, "Number of kernel size should be one less than number of channels."
        self.k_list, self.c_list, self.p_list = k_list, c_list, p_list
        if s_list is None:
            self.s_list = [1] * (len(c_list) - 1)
        else:
            self.s_list = s_list
        
        if p_list is None:
            self.p_list = [k // 2 for k in k_list]
        else:
            self.p_list = p_list
    
        layers = []

        for i in range(len(self.c_list) - 2):
            layers.append(nn.Conv2d(self.c_list[i], self.c_list[i+1], self.k_list[i], \
                stride=self.s_list[i], padding=self.p_list[i]))
            layers.append(nn.BatchNorm2d(self.c_list[i+1]))
            layers.append(nn.GELU())

        layers.append(nn.Conv2d(self.c_list[-2], self.c_list[-1], self.k_list[-1], \
                                      stride=self.s_list[-1], padding=self.p_list[-1]))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).exp()


class LinearCovHead(nn.Module):
    """
    Estimate the covariance using a linear feed-forward architecture.
    """
    def __init__(self, c_list: List[int] = [128, 32, 2] ):
        super().__init__()
        layers = []
        for i in range(len(c_list)-2):
            layers.append(nn.Linear(c_list[i], c_list[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(c_list[-2], c_list[-1]))

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        The feature input is default as [B, C, H, W], given that most of the encoder are
        CNN or transformer based.
        """
        b, c, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)
        cov = self.layers(x).exp()
        return cov.permute(0, 2, 1).reshape(b, 2, h, w)


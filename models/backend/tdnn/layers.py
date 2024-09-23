import math
import torch
import torch.nn as nn


class SEModule(nn.Module):
    def __init__(self, channels: int, bottleneck=128):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(
                in_channels  = channels, 
                out_channels = bottleneck, 
                kernel_size  = 1, 
                padding      = 0
            ),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(
                in_channels  = bottleneck, 
                out_channels = channels, 
                kernel_size  = 1, 
                padding      = 0
            ),
            nn.Sigmoid(),
            )

    def forward(self, input):
        x = self.se(input)
        return input * x
    

class Res2Conv1D(nn.Module):
    """ Res2Conv1D """
    def __init__(self, dim, kernel_size, dilation, scale=8):
        super().__init__()
        width      = int(math.floor(dim / scale))
        self.nums  = scale -1
        convs      = []
        bns        = []
        num_pad    = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(
                in_channels  = width, 
                out_channels = width, 
                kernel_size  = kernel_size, 
                dilation     = dilation, 
                padding      = num_pad
            ))
            bns.append(nn.BatchNorm1d(num_features=width))
        self.convs = nn.ModuleList(convs)
        self.bns   = nn.ModuleList(bns)
        self.act   = nn.ReLU()
        self.width = width
        
    def forward(self, x):
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i==0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.act(sp)
            sp = self.bns[i](sp)
            if i==0:
                x = sp
            else:
                x = torch.cat((x, sp), 1)
        x = torch.cat((x, spx[self.nums]),1)  
        return x

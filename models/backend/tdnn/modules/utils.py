import torch
import torch.nn as nn
import torch.nn.functional as F


class Sparse(nn.Module):
    def __init__(self, sparse=0.1):
        super().__init__()
        assert 0 <= sparse <= 1
        self.sparse = sparse
        
    def forward(self, x): # x=[B, C, T//2+1]
        if self.sparse == 1:
            return torch.zeros_like(x).to(x.device)
    
        elif self.sparse == 0:
            return x
        else:
            mask   = (torch.rand(x.shape[:-1]) > self.sparse).float().to(x.device)
            mask_f = mask.unsqueeze(-1).expand(x.shape)
            
        return mask_f, (1.0-mask_f)*(x.abs().mean())


class attn_fn(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attn_fn, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K # K//16
        self.fc1 = nn.Conv1d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv1d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x/self.temperature, 1)


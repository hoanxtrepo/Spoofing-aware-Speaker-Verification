'''
This is the ECAPA-TDNN model.
This model is modified and combined based on the following three projects:
  1. https://github.com/clovaai/voxceleb_trainer/issues/86
  2. https://github.com/lawlict/ECAPA-TDNN/blob/master/ecapa_tdnn.py
  3. https://github.com/speechbrain/speechbrain/blob/96077e9a1afff89d3f5ff47cab4bca0202770e4f/speechbrain/lobes/models/ECAPA_TDNN.py

'''

import math
from typing import Dict

import torch
import torch.nn as nn

from .layers import SEModule
from .augment import FbankAug


class Bottle2neck(nn.Module):

    def __init__(self, inplanes: int, planes: int, kernel_size=None, dilation=None, scale = 8):
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(
            in_channels  = inplanes, 
            out_channels = width * scale, 
            kernel_size  = 1
        )
        self.bn1    = nn.BatchNorm1d(num_features=width * scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size / 2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(
                in_channels  = width, 
                out_channels = width, 
                kernel_size  = kernel_size, 
                dilation     = dilation, 
                padding      = num_pad
            ))
            bns.append(nn.BatchNorm1d(num_features=width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(
            in_channels  = width*scale, 
            out_channels = planes, 
            kernel_size  = 1
        )
        self.bn3    = nn.BatchNorm1d(num_features=planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(channels=planes)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            sp = spx[i] if i==0 else sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            out = sp if i==0 else torch.cat((out, sp), 1)

        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)

        out = self.se(out)

        return out +residual


class ECAPA_TDNN(nn.Module):

    def __init__(self, idims: int, hparams: Dict):

        super(ECAPA_TDNN, self).__init__()

        # initialize hparams
        self.hparams   = hparams
        self.specaug   = FbankAug() # Spec augmentation

        # initialize model structure
        self.idims  = idims
        self.odims  = self.hparams["embedding_size"]

        self.conv1  = nn.Conv1d(
           in_channels  = self.idims, 
           out_channels = self.hparams["C"], 
           kernel_size  = self.hparams["kernel_size"], 
           stride       = 1, 
           padding      = 2
        )
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(num_features=self.hparams["C"])
        self.layer1 = Bottle2neck(
           inplanes    = self.hparams["C"], 
           planes      = self.hparams["C"], 
           kernel_size = self.hparams["bottle_neck"]["kernel_size"], 
           dilation    = self.hparams["bottle_neck"]["dilation"][0], 
           scale       = self.hparams["bottle_neck"]["scale"]
        )
        self.layer2 = Bottle2neck(
           inplanes    = self.hparams["C"], 
           planes      = self.hparams["C"], 
           kernel_size = self.hparams["bottle_neck"]["kernel_size"], 
           dilation    = self.hparams["bottle_neck"]["dilation"][1], 
           scale       = self.hparams["bottle_neck"]["scale"]
        )
        self.layer3 = Bottle2neck(
           inplanes    = self.hparams["C"], 
           planes      = self.hparams["C"], 
           kernel_size = self.hparams["bottle_neck"]["kernel_size"], 
           dilation    = self.hparams["bottle_neck"]["dilation"][2], 
           scale       = self.hparams["bottle_neck"]["scale"]
        )
        
        # fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(
           in_channels  = 3 * self.hparams["C"], 
           out_channels = self.hparams["fixed_C"], 
           kernel_size  = 1
        )
        self.attention = nn.Sequential(
            nn.Conv1d(
                in_channels  = self.hparams["fixed_C"] * 3, 
                out_channels = self.hparams["attn_dims"], 
                kernel_size  = 1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.hparams["attn_dims"]),
            nn.Tanh(), # add this layer
            nn.Conv1d(
                in_channels  = self.hparams["attn_dims"], 
                out_channels = self.hparams["fixed_C"], 
                kernel_size  = 1
            ),
            nn.Softmax(dim=2),
        )
        self.bn5 = nn.BatchNorm1d(num_features=self.hparams["fixed_C"] * 2)
        self.fc6 = nn.Linear(
           in_features  = self.hparams["fixed_C"] * 2, 
           out_features = self.odims
        )
        self.bn6 = nn.BatchNorm1d(num_features=self.odims)

    def forward(self, x: torch.Tensor, aug: bool):
        """Calculate forward propagation.

        Args:
            x (Tensor): Feature tensor (B, T_feats, aux_channels).
            aug (bool): Use spec augmentation
        """

        if aug == True:
            x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)
        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2)*w, dim=2) - mu**2).clamp(min=1e-4))

        x = torch.cat((mu, sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

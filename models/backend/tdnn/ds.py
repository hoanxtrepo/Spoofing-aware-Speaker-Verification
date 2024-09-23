import math
from typing import Dict

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

from .modules import SparseDGF
from .modules.weight_init import trunc_normal_
from .layers import Res2Conv1D, SEModule
from .augment import FbankAug


class LocalBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, dilation=1, scale=4, drop_path=0.):
        super().__init__()
        self.res2conv = Res2Conv1D(dim, kernel_size, dilation, scale)     
           
        self.norm1 = nn.BatchNorm1d(dim)   
        self.norm2 = nn.BatchNorm1d(dim)   
        self.norm3 = nn.BatchNorm1d(dim)   
        self.proj1 = nn.Conv1d(dim, dim, kernel_size=1)  
        self.proj2 = nn.Conv1d(dim, dim, kernel_size=1)
        self.act   = nn.ReLU()
        # self.act = nn.GELU()
        self.se    = SEModule(dim)

    def forward(self, x):
        skip = x
        
        x = self.proj1(x)
        x = self.act(x)
        x = self.norm1(x)
        
        x = self.res2conv(x)
        # x = self.dwconv(x)
        # x = self.act(x)
        # x = self.norm2(x)
  
        x = self.proj2(x)
        x = self.act(x)
        x = self.norm3(x)
        
        x = skip + self.se(x)
        
        return x    


class GlobalBlock(nn.Module):
    """ 
     Global block: if global modules = MSA or LSTM, need to permute the dimension of input tokens
    """
    def __init__(self, dim, T=200, dropout=0.2, K=4):
        super().__init__()
        self.gf = SparseDGF(dim, T, dropout=dropout, K=K) # Dynamic global-aware filters with sparse regularization
        # self.gf = SparseGF(dim, T, dropout=dropout) # Global-aware filters with sparse regularization
        # self.gf = DGF(dim, T, K=K) # Dynamic global-aware filters
        # self.gf = GF(dim, T) # Global-aware filters
        # self.gf = MSA(num_attention_heads=K, input_size=dim, hidden_size=dim) # Multi-head self-attention
        # self.gf = LSTM(input_size=dim, hidden_size=dim, bias=False, bidirectional=False) # LSTM
        
        self.norm1 = nn.BatchNorm1d(dim)  
        self.norm2 = nn.BatchNorm1d(dim)  
        self.norm3 = nn.BatchNorm1d(dim)  
        self.proj1 = nn.Conv1d(dim, dim, kernel_size=1)  
        self.proj2 = nn.Conv1d(dim, dim, kernel_size=1)  
        self.act   = nn.ReLU()

    def forward_(self, x):
        skip = x
        
        x = self.proj1(x)
        x = self.act(x)
        x = self.norm1(x)
        
        x = self.gf.forward_(x) 
        x = self.act(x)    
        x = self.norm2(x)
        
        x = self.proj2(x)
        x = self.act(x) 
        x = self.norm3(x) 
        
        x = skip + x
        
        return x    
    
    def forward(self, x):
        skip = x
        
        x = self.proj1(x)
        x = self.act(x)
        x = self.norm1(x) 
        
        # if self.gf == SA or LSTM, we need to transform the input tokens x
        # x = x.permute(0,2,1)
        x = self.gf(x) 
        # x = x.permute(0,2,1)
       
        x = self.act(x)      
        x = self.norm2(x)
        
        x = self.proj2(x)
        x = self.act(x) 
        x = self.norm3(x) 
        
        x = skip + x
        
        return x    


class DS_TDNN(nn.Module):

    def __init__(self, idims: int, n_classes: int, hparams: Dict):

        super(DS_TDNN, self).__init__()

        # initialize hparams
        self.n_classes = n_classes
        self.hparams   = hparams
        self.sparse    = True
        self.specaug   = FbankAug() # Spec augmentation

        # initialize model structure
        self.idims  = idims
        self.odims  = self.hparams["embedding_size"]

        self.conv1  = nn.Conv1d(
            in_channels  = self.idims,
            out_channels = self.hparams["C"],
            kernel_size  = 5, 
            stride       = 1, 
            padding      = 2
        )
        self.relu   = nn.ReLU()
        self.gelu   = nn.GELU()
        self.bn1    = nn.BatchNorm1d(num_features=self.hparams["C"])
        
        # initialize local branch
        self.llayer1 = LocalBlock(
            dim         = self.hparams["C"] // 2, 
            kernel_size = self.hparams["local_block"]["kernel_size"][0], 
            scale       = self.hparams["local_block"]["scale"][0]
        ) 

        # NOTE: kernel size: 8 8 8 self.hparams["C"] = 1024; 4 6 8 self.hparams["C"]=960
        self.llayer2 = LocalBlock(
            dim         = self.hparams["C"] // 2, 
            kernel_size = self.hparams["local_block"]["kernel_size"][1], 
            scale       = self.hparams["local_block"]["scale"][1]
        )
        self.llayer3 = LocalBlock(
            dim         = self.hparams["C"] // 2, 
            kernel_size = self.hparams["local_block"]["kernel_size"][2],
            scale       = self.hparams["local_block"]["scale"][2]
        )

        # initialize global branch
        self.glayer1 = GlobalBlock(
            dim     = self.hparams["C"] // 2, 
            T       = self.hparams["global_block"]["T"], 
            dropout = self.hparams["global_block"]["drop_out"][0],  
            K       = self.hparams["global_block"]["K"][0]
        )
        self.glayer2 = GlobalBlock(
            dim     = self.hparams["C"] // 2, 
            T       = self.hparams["global_block"]["T"], 
            dropout = self.hparams["global_block"]["drop_out"][0],  
            K       = self.hparams["global_block"]["K"][0]
        )
        self.glayer3 = GlobalBlock(
            dim     = self.hparams["C"] // 2, 
            T       = self.hparams["global_block"]["T"], 
            dropout = self.hparams["global_block"]["drop_out"][0],  
            K       = self.hparams["global_block"]["K"][0]
        )

        self.layer4 = nn.Conv1d(
            in_channels  = 3 * self.hparams["C"], 
            out_channels = self.hparams["fixed_C"], 
            kernel_size  = 1
        )
        
        # ASP
        self.attention = nn.Sequential(
            nn.Conv1d(
                in_channels  = self.hparams["fixed_C"] * 3, 
                out_channels = self.hparams["attn_dims"], 
                kernel_size  = 1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=self.hparams["attn_dims"]),
            nn.Tanh(), # I add this layer
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
        self.uniform_init = self.hparams["uniform_init"]

    def _forward_sparse(self, x: torch.Tensor, aug: bool=None):    
        """Calculate sparse forward propagation.

        Args:
            x (Tensor): Feature tensor (B, T_feats, aux_channels).
            aug (bool): Use spec augmentation
        """
        
        assert self.sparse==True
        if aug == True:
            x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        lx, gx = torch.chunk(x, 2, dim=1)

        # Dual branch
        lx1 = self.llayer1(lx)
        gx1 = self.glayer1.forward_(gx)
 
        lx2 = self.llayer2(0.8*lx1+0.2*gx1)
        gx2 = self.glayer2.forward_(0.8*gx1+0.2*lx1)

        lx3 = self.llayer3(0.8*lx2+0.2*gx2)
        gx3 = self.glayer3.forward_(0.8*gx2+0.2*lx2)    
        
        x = self.layer4(torch.cat((lx1, gx1, lx2, gx2, lx3, gx3),dim=1))
        x = self.relu(x)
        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)

        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt((torch.sum((x**2)*w, dim=2) - mu**2).clamp(min=1e-4))

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x
    
    def _forward_no_sparse(self, x: torch.Tensor, aug: bool=None):
        """Calculate no sparse forward propagation.

        Args:
            x (Tensor): Feature tensor (B, T_feats, aux_channels).
            aug (bool): Use spec augmentation
        """


        if aug == True:
            x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        lx, gx = torch.chunk(x, 2, dim=1)
        
        # Dual branch:
        lx1 = self.llayer1(lx)
        gx1 = self.glayer1(gx)
                
        lx2 = self.llayer2(0.8*lx1+0.2*gx1)
        gx2 = self.glayer2(0.8*gx1+0.2*lx1)
        
        lx3 = self.llayer3(0.8*lx2+0.2*gx2)
        gx3 = self.glayer3(0.8*gx2+0.2*lx2)   

        x = self.layer4(torch.cat((lx1,gx1, lx2,gx2, lx3,gx3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x

    def forward(self, x: torch.Tensor, aug: bool):
        """Calculate forward propagation.

        Args:
            x (Tensor): Feature tensor (B, T_feats, aux_channels).
            aug (bool): Use spec augmentation
        """
        if self.sparse is True:
            
            return self._forward_sparse(x, aug)
        else:

            return self._forward_no_sparse(x, aug)

    def hook(self, x: torch.Tensor, aug: bool=None):
        '''
         hook for different-scale feature maps
        '''
        with torch.no_grad():
            x = self.torchfbank(x)+1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if aug == True:
                x = self.specaug(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)
        
        stem_o = x
        
        lx, gx = torch.chunk(x, 2, dim=1)
        
        #Dual branch:
        lx1 = self.llayer1(lx)
        gx1 = self.glayer1(gx)
                
        lx2 = self.llayer2(0.8*lx1+0.2*gx1)
        gx2 = self.glayer2(0.8*gx1+0.2*lx1)
        
        lx3 = self.llayer3(0.8*lx2+0.2*gx2)
        gx3 = self.glayer3(0.8*gx2+0.2*lx2)        

        x = self.layer4(torch.cat((lx1,gx1, lx2,gx2, lx3,gx3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)

        return x, stem_o, [lx, lx1, lx2, lx3], [gx, gx1, gx2, gx3], [lx1+gx1, lx2+gx2]
        
    def _init_weights(self, m):
        if not self.uniform_init:
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)     

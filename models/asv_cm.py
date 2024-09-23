from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .asv import ASVClassifier
from .cm import CMClassifier

class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        #channel wise
        self.fc1 = nn.Linear(in_features=256, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=256)
        self.relu = nn.LeakyReLU(negative_slope=0.3)
        self.sigmoid = nn.Sigmoid()
        
        #feature wise
        
    def channel_wise(self,emb):
        emb = F.adaptive_avg_pool1d(emb, output_size=1) #bs,256,1
        emb= emb.view(emb.shape[0], 1,256) #bs,1,256
        emb = self.fc1(emb) #bs,1,64
        emb= emb.view(emb.shape[0], 64,1) #,bs,64,1
        emb= self.relu(emb) #,bs,64,1
        
        emb= emb.view(emb.shape[0], 1,64) #bs,1,64
        emb= self.fc2(emb) #bs,1,256
        emb= emb.view(emb.shape[0], 256,1) #bs,256,1
        emb = self.sigmoid(emb)
        return emb 
    def feature_wise(self, emb):
        emb = torch.transpose(emb,1,2) #bs,256,256
        emb = F.adaptive_avg_pool1d(emb, output_size=1) #bs,256,1
        emb= emb.view(emb.shape[0], 1,256) #bs,1,256
        emb = self.fc1(emb) #bs,1,64
        emb= emb.view(emb.shape[0], 64,1) #,bs,64,1
        emb= self.relu(emb) #,bs,64,1
        
        emb= emb.view(emb.shape[0], 1,64) #bs,1,64
        emb= self.fc2(emb) #bs,1,256
        emb= emb.view(emb.shape[0], 256,1) #bs,256,1
        emb = torch.transpose(emb,1,2) #bs,1,256
        emb = self.sigmoid(emb)
        return emb

class ASVCMClassifier(nn.Module):
	def __init__(self, asv_hparams: str, cm_hparams: str, fixed: bool=False):

		super().__init__()

		# initialize backbone
		self.asv_hparams = asv_hparams
		self.asv_emb = ASVClassifier(
			n_class  = 1, 
			hparams  = self.asv_hparams["hparams"],
			frontend_type = self.asv_hparams["model"]["frontend"],
			backend_type  = self.asv_hparams["model"]["backend"]
		)
		
		self.cm_hparams = cm_hparams
		self.cm_emb  = CMClassifier(
			n_class  = 1, 
			hparams  = self.cm_hparams["hparams"],
			frontend_type = self.cm_hparams["model"]["frontend"],
			backend_type  = self.cm_hparams["model"]["backend"]
		)
		self.fixed = fixed
		if fixed is True:
			for layer in [self.asv_emb, self.cm_emb]:
				for param in layer.parameters():
					param.requires_grad = False

		# initialize pre-process input
		self.asv_dim = self.asv_emb.speaker_encoder.odims
		self.cm_dim  = self.cm_emb.speaker_encoder.odims

		# initialize linear layers
		self.asv_fc = nn.Sequential(
			nn.Linear(in_features=self.asv_dim, out_features=256),
			nn.LeakyReLU(negative_slope=0.3),
		)
		self.cm_fc = nn.Sequential(
			nn.Linear(in_features=self.cm_dim, out_features=256),
			nn.LeakyReLU(negative_slope=0.3),
		)

		# initialize DNN layer
		conv_channels = [64, 128, 256]
		self.conv1 = nn.Conv1d(in_channels=3, out_channels=conv_channels[0], bias=False, kernel_size=3, padding=1)
		self.conv2 = nn.Conv1d(in_channels=64, out_channels=conv_channels[1], bias=False, kernel_size=3, padding=1)
		self.conv3 = nn.Conv1d(in_channels=128, out_channels=conv_channels[2], bias=False, kernel_size=3, padding=1)

		self.bn1 = nn.BatchNorm1d(num_features=conv_channels[0])
		self.bn2 = nn.BatchNorm1d(num_features=conv_channels[1])
		self.bn3 = nn.BatchNorm1d(num_features=conv_channels[2])

		self.relu = nn.LeakyReLU(negative_slope=0.3)
		
		# initialize classifier layer
		self.DNN_layer = nn.Sequential(
			nn.Linear(in_features=1024, out_features=512),
			nn.LeakyReLU(negative_slope=0.3),
			nn.Linear(in_features=512, out_features=256),
			nn.LeakyReLU(negative_slope=0.3),
		)
		self.fc_out = nn.Linear(in_features=256, out_features=1, bias=False)
  
		#MY CODE HERE
		#initialize global attention
		self.attention = Attention()
  
	def save_parameters(self, path: str):
		torch.save(self.state_dict(), path)

	def load_parameters(self, path: str):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			origname = name
			if name not in self_state:
				name = name.replace("module.", "")
				if name not in self_state:
					print("%s is not in the model."%origname)
					continue
			if self_state[name].size() != loaded_state[origname].size():
				print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
				continue
			self_state[name].copy_(param)

	def forward(self, asv_enr_wav: torch.Tensor, asv_tst_wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
		asv_enr_emb = self.asv_emb._forward(asv_enr_wav, aug=True)   # shape: (bs, 192)
		asv_enr_emb = self.asv_fc(asv_enr_emb)						 # shape: (bs, 256)
		asv_tst_emb = self.asv_emb._forward(asv_tst_wav, aug=True)   # shape: (bs, 192)
		asv_tst_emb = self.asv_fc(asv_tst_emb)						 # shape: (bs, 256)

		spf_emb = self.cm_emb._forward(asv_tst_wav, aug=True) # shape: (bs, 160)
		spf_emb = self.cm_fc(spf_emb) # shape: (bs, 256)

		embs = torch.stack([asv_enr_emb, asv_tst_emb, spf_emb], dim=1) # shape: (bs, 3, 256)

		embs = self.relu(self.bn1(self.conv1(embs))) # shape: (bs, 64, 256)
		embs = self.relu(self.bn2(self.conv2(embs))) # shape: (bs, 128, 256)
		embs = self.relu(self.bn3(self.conv3(embs))) # shape: (bs, 256, 256)
        # add attention layer here
		# channel_emb = self.attention.channel_wise(embs)
		# feature_emb = self.attention.feature_wise(embs)
		# embs = channel_emb*embs*feature_emb
        
		embs = F.adaptive_avg_pool1d(embs, output_size=4) # shape: (bs, 256, 4)
		embs = torch.flatten(embs, start_dim=1) # shape: (bs, 1024)		
        
		embs = self.DNN_layer(embs)
		outs = self.fc_out(embs).squeeze(1)
		
		return embs, outs

	def validate(
			self, 
			asv_enr_wav: torch.Tensor=None, 
			asv_tst_wav: torch.Tensor=None,
			asv_enr_emb: torch.Tensor=None,
			asv_tst_emb: torch.Tensor=None,
			spf_emb: torch.Tensor=None
		) -> Tuple[torch.Tensor, torch.Tensor]:

		if asv_enr_emb is None:
			assert asv_enr_wav is not None
			asv_enr_emb = self.asv_emb._forward(asv_enr_wav)   # shape: (bs, 192)
		
		if asv_tst_emb is None:
			assert asv_tst_wav is not None
			asv_tst_emb = self.asv_emb._forward(asv_tst_wav)   # shape: (bs, 192)

		if spf_emb is None:
			assert asv_tst_wav is not None
			spf_emb = self.cm_emb._forward(asv_tst_wav) # shape: (bs, 160)
		
		asv_enr_emb = self.asv_fc(asv_enr_emb) # shape: (bs, 256)
		asv_tst_emb = self.asv_fc(asv_tst_emb) # shape: (bs, 256)
		spf_emb 	= self.cm_fc(spf_emb) 		   # shape: (bs, 256)
     
		embs = torch.stack([asv_enr_emb, asv_tst_emb, spf_emb], dim=1) # shape: (bs, 3, 256)

		embs = self.relu(self.bn1(self.conv1(embs))) # shape: (bs, 64, 256)
		embs = self.relu(self.bn2(self.conv2(embs))) # shape: (bs, 128, 256)
		embs = self.relu(self.bn3(self.conv3(embs))) # shape: (bs, 256, 256)

		embs = F.adaptive_avg_pool1d(embs, output_size=4) # shape: (bs, 256, 4)
		embs = torch.flatten(embs, start_dim=1) # shape: (bs, 1024)		
        
		embs = self.DNN_layer(embs)
		outs = self.fc_out(embs).squeeze(1)

		return embs, outs

import sys
import tqdm
import time
import librosa
from sklearn.metrics import f1_score
from typing import Dict
import soundfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from speechbrain.lobes.models.Xvector import Discriminator as Classifier
from .frontend.pre_emphasis import PreEmphasis


class CMClassifier(nn.Module):
	def __init__(
		self, 
		n_class: int,
		hparams: Dict,
		frontend_type: str="mel_spectrogram",
		backend_type: str="ecapa_tdnn"
	):
		super(CMClassifier, self).__init__()
		self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.hparams = hparams
		self.n_class = n_class
		self.sr		 = hparams["preprocessing"]["signal"]["sampling_rate"]
		self.max_length = hparams["preprocessing"]["num_frames"] * 160 + 240
		
		## initialize fronend
		self.frontend_type = frontend_type
		if self.frontend_type == "none":
			self.feat_extractor = PreEmphasis()
		else:
			raise NotImplementedError(f"Frontend model {self.frontend_type} not implemented yet!")
		self.feat_extractor.to(self.device)

		## initialize backend
		self.backend_type = backend_type
		if self.backend_type == "assist":
			from .backend.assist import ASSIST
			self.speaker_encoder = ASSIST(d_args=self.hparams["assist"])
			if self.frontend_type == "sinc_net": 
				# NOTE(ducnt2): SincNet take input is data like Waveform
				self.feat_extractor.flatten = True
		else:
			raise NotImplementedError(f"Not implemented backend {self.backend_type} yet!")
		self.speaker_encoder.to(self.device)
		
		# initialize classifier
		self.classifier = Classifier(
			input_shape = [None, None, self.speaker_encoder.odims],
			lin_blocks  = 1,
			lin_neurons = 512,
			out_neurons = self.n_class,
		).to(self.device)

		## model classifier & loss function
		self.speaker_loss = nn.BCEWithLogitsLoss()

		# optimize & scheduler 
		self.optimizer = optim.Adam(self.parameters(), lr=0.0001, weight_decay=0.000002)
		self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, factor=0.8, patience=2, cooldown=5)

		self.n_params = sum(param.numel() for param in self.parameters())
		print(time.strftime("%m-%d %H:%M:%S") + f" Model parameter number {round(self.n_params / 1024 / 1024, 2)}M && {n_class} speakers")

	def _forward(self, datas: torch.Tensor, aug: bool=False) -> torch.Tensor:
		feats  = self.feat_extractor(datas) if self.speaker_encoder is not None else datas
		embeds = self.speaker_encoder(feats, aug=aug)
		
		return embeds

	def train_network(self, epoch: int, loader: DataLoader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optimizer.param_groups[0]["lr"]
		for num, (datas, labels) in enumerate(loader, start=1):
			labels   = labels.to(self.device).float()
			spk_embs = self._forward(datas=datas.to(self.device), aug=True)
			preds    = self.classifier(spk_embs).squeeze()
			nloss    = self.speaker_loss(preds, labels)

			self.optimizer.zero_grad()
			nloss.backward()
			self.optimizer.step()
			
			index += len(labels)
			top1  += (torch.sigmoid(preds).round() == labels).float().mean() * 100
			loss  += nloss.detach().cpu().numpy()
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "	%(epoch, lr, 100 * (num / loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"		%(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()

		sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)

	def eval_network(self, eval_list: str):
		self.eval()
		lines = [line.split("|") for line in open(eval_list).read().split("\n") if line]
		
		predictions, targets = [], []
		for line in tqdm.tqdm(lines[: 50], leave=False):
			filename, label = line
			x = soundfile.read(filename)
			with torch.no_grad():
				y_pred = self.classifier(self._forward(torch.FloatTensor(x).unsqueeze(0).to(self.device))).squeeze()
			predictions.append(torch.sigmoid(y_pred).round().item())
			targets.append(float(label))

		# Coumpute EER and minDCF
		score = round(f1_score(targets, predictions, average="weighted"), 6) * 100
		
		return (score, )

	def infer(self, eval_list:str, save_path):
		count_fail = 0
		self.eval()
		scores = []
		failed_lines = []
		paths = [line.split("\t")[1] for line in open(eval_list).read().split("\n") if line]
		for path in tqdm.tqdm(paths, leave= False):
			try:
				x,_ = soundfile.read(path)
				with torch.no_grad():
					x = torch.FloatTensor(x)
					x = x.unsqueeze(0)
					x = self._forward(x.to(self.device))
					y_pred = self.classifier(x).squeeze()
				score = torch.sigmoid(y_pred).item()
				scores.append(score)

			except:
				count_fail += 1
				# failed_lines.append(path)
				score = -0.5
				scores.append(score)
				print('failed')
				continue
			# scores.append(score)
		with open(save_path, "w") as f:
			for score in scores:
				f.write(str(score)+"\n")
		# with open(save_path.replace(".txt", "_failed.txt"), "w") as f:
		# 	for line in failed_lines:
		# 		f.write("\t".join(line) + "\n")
		# print("number of failed audios: ", count_fail)


		# with open(eval_list, "r") as f:
		# 	lines = f.readlines()
		# for line in tqdm.tqdm(lines, leave=False):
		# 	path = line.split("\t")[1]
		# 	# path = paths[:5]
		# 	try:
		# 		x,_ = soundfile.read(path)
		# 		with torch.no_grad():
		# 			x = torch.FloatTensor(x)
		# 			x = x.unsqueeze(0)
		# 			x = self._forward(x.to(self.device))
		# 			y_pred = self.classifier(x).squeeze()
		# 	except:
		# 		failed_lines.append(line)
		# 		count_fail += 1
		# 		continue
		# with open(save_path, "w") as f:
		# 	for score in scores:
		# 		f.write(str(score)+"\n")
		
		# with open(save_path.replace(".txt", "_failed.txt"), "w") as f:
		# 	for line in failed_lines:
		# 		f.write("\t".join(line) + "\n")

		# print("number of failed audios: ", count_fail)
		return 0
   
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

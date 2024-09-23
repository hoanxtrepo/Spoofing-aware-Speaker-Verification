import sys
import tqdm
import time
import math
import librosa
import numpy as np
from typing import Dict
import soundfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.tools.utils import tuneThresholdfromScore, ComputeErrorRates, ComputeMinDcf
from .frontend.pre_emphasis import PreEmphasis


class ASVClassifier(nn.Module):
	def __init__(
		self, 
		n_class: int,
		hparams: Dict,
		frontend_type: str="mel_spectrogram",
		backend_type: str="ecapa_tdnn"
	):
		super(ASVClassifier, self).__init__()
		self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		self.hparams = hparams
		self.sr		 = hparams["preprocessing"]["signal"]["sampling_rate"]
		self.max_length = hparams["preprocessing"]["num_frames"] * 160 + 240
		
		## fronend model
		self.frontend_type = frontend_type
		if self.frontend_type == "mel_spectrogram":
			from torchaudio.transforms import MelSpectrogram
			self.feat_extractor = nn.Sequential(
				PreEmphasis(),
				MelSpectrogram(
					sample_rate = self.hparams["preprocessing"]["signal"]["sampling_rate"],
					n_fft       = self.hparams["preprocessing"]["stft"]["filter_length"], 
					win_length  = self.hparams["preprocessing"]["stft"]["win_length"], 
					hop_length  = self.hparams["preprocessing"]["stft"]["hop_length"], 
					f_max       = self.hparams["preprocessing"]["mel"]["fmax"], 
					f_min       = self.hparams["preprocessing"]["mel"]["fmin"], 
					window_fn   = torch.hamming_window, 
					n_mels      = self.hparams["preprocessing"]["mel"]["channels"]
				)
			)
			self.feat_dims  = self.hparams["preprocessing"]["mel"]["channels"]
		elif self.frontend_type == "sinc_net":
			from .frontend.sinc_net import SincNet
			self.fs		= self.hparams["preprocessing"]["signal"]["sampling_rate"]
			self.feat_extractor = SincNet(sample_rate = self.fs)
			self.feat_dims  	= self.feat_extractor.out_dim
		else:
			raise NotImplementedError(f"Not implemented frontend {self.frontend_type} yet!")
		self.feat_extractor.to(self.device)

		## backend model
		self.backend_type = backend_type
		if self.backend_type == "ecapa_tdnn":
			from .backend.tdnn import ECAPA_TDNN
			self.speaker_encoder = ECAPA_TDNN(
				idims 	  = self.feat_dims,
				hparams   = hparams["ecapa_tdnn"]
			)
		elif self.backend_type == "ds_tdnn":
			from .backend.tdnn import DS_TDNN
			self.speaker_encoder = DS_TDNN(
				idims 	  = self.feat_dims,
				n_classes = n_class,
				hparams   = hparams["ds_tdnn"]
			)
		else:
			raise NotImplementedError(f"Not implemented backend {self.backend_type} yet!")
		self.speaker_encoder.to(self.device)
		
		## model classifier & loss function
		self.loss_func = self.hparams["loss_func"]
		if self.loss_func == "aam_softmax":
			from .loss import AAMsoftmax as LossFunction
		elif self.loss_func == "softmax":
			from .loss import Softmax as LossFunction
		elif self.loss_func == "nnnloss":
			from .loss import NNNLoss as LossFunction
		else:
			raise NotImplementedError(f"Not implemented loss function {self.loss_func} yet!")
		self.speaker_loss = LossFunction(
			nOut	 = self.speaker_encoder.odims,
			nClasses = n_class, 
			**hparams["loss"]
		).to(self.device)

		# optimize & scheduler 
		self.optimizer = optim.Adam(
			params 		 = self.parameters(), 
			lr			 = hparams["optim"]["lr"], 
			weight_decay = hparams["optim"]["wd"]
		)
		self.scheduler = optim.lr_scheduler.StepLR(
			optimizer = self.optimizer, 
			step_size = 1,
			gamma	  = hparams["optim"]["lr_decay"]
		)

		self.n_params = sum(param.numel() for param in self.parameters())
		print(time.strftime("%m-%d %H:%M:%S") + f" Model parameter number {round(self.n_params / 1024 / 1024, 2)}M && {n_class} speakers")

	def _forward(self, datas: torch.Tensor, aug: bool=False) -> torch.Tensor:
		if self.frontend_type == "mel_spectrogram":	
			feats = self.feat_extractor(datas) + 1e-6
			feats = feats.log()
			feats = feats - torch.mean(feats, dim=-1, keepdim=True) # NOTE: input normalization
		elif self.frontend_type.startswith("sinc_net"):
			feats = self.feat_extractor(datas)
		
		embeds = self.speaker_encoder(feats, aug=aug)
		return embeds

	def _extract_feature(self, filepath: str):
		if isinstance(filepath, str):
			audio, _ = soundfile.read(filepath)
		else:
			audio = filepath

		# full utterance
		if self.frontend_type == "mel_spectrogram":
			data_1 = torch.FloatTensor(np.stack([audio],axis=0)).to(self.device)
		else:
			bs = math.ceil(audio.shape[0] / self.max_length)
			data_1 = torch.zeros(bs, self.max_length)
			i, s, e = 0, 0, self.max_length
			while s < audio.shape[0]:
				_data = torch.FloatTensor(audio[s: e])
				data_1[i][: _data.size(0)] = _data
				i += 1
				s += self.max_length
				e += self.max_length
			data_1 = data_1.to(self.device)

		# splited utterance
		if audio.shape[0] <= self.max_length:
			shortage = self.max_length - audio.shape[0]
			audio = np.pad(audio, (0, shortage), "wrap")
		feats = []
		startframe = np.linspace(0, audio.shape[0] - self.max_length, num=5)
		for asf in startframe:
			feats.append(audio[int(asf) :int(asf) + self.max_length])
		feats = np.stack(feats, axis = 0).astype(float)
		data_2 = torch.FloatTensor(feats).to(self.device)

		return (data_1, data_2)

	def train_network(self, epoch: int, loader: DataLoader):
		self.train()
		## Update the learning rate based on the current epcoh
		self.scheduler.step(epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optimizer.param_groups[0]["lr"]
		for num, (data, labels) in enumerate(loader, start=1):
			self.optimizer.zero_grad()
			speaker_embedding = self._forward(datas=data.to(self.device), aug=True)
			nloss, prec       = self.speaker_loss(speaker_embedding, labels.to(self.device))
			nloss.backward()
			self.optimizer.step()
			index += len(labels)
			top1  += prec
			loss  += nloss.detach().cpu().numpy()
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "    %(epoch, lr, 100 * (num / loader.__len__())) + \
			" Loss: %.5f, ACC: %2.2f%% \r"        %(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
		sys.stdout.write("\n")
		return loss/num, lr, top1/index*len(labels)

	def eval_network(self, eval_list: str):
		self.eval()
		files = []
		embeddings = {}
		# lines = open(eval_list).read().splitlines()
		lines = [line.split() for line in open(eval_list).read().split("\n") if line]
		for line in lines:
			files.append(line[1])
			files.append(line[2])
		setfiles = list(set(files))
		setfiles.sort()

		for file in tqdm.tqdm(setfiles, total=len(setfiles)):
			data_1, data_2 = self._extract_feature(file)
			
			# speaker embeddings
			with torch.no_grad():
				embedding_1 = self._forward(data_1, aug=False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1).mean(dim=0, keepdim=True)
				embedding_2 = self._forward(data_2, aug=False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1).mean(dim=0, keepdim=True)
			embeddings[file] = [embedding_1, embedding_2]
		scores, labels  = [], []

		for line in lines:
			embedding_11, embedding_12 = embeddings[line[2]]
			embedding_21, embedding_22 = embeddings[line[1]]
			
			# compute the scores
			score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
			score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
			score = (score_1 + score_2) / 2
			score = score.detach().cpu().numpy()
			scores.append(score)
			if len(line) == 3:
				labels.append(int(line[0]))
		
		if len(labels) == 0:
			reported_datas = []
			for i in range(len(scores)):
				reported_datas.append(f"{'/'.join(lines[i][0].split('/')[-2: ])}\t{'/'.join(lines[i][1].split('/')[-2: ])}\t{scores[i]}")
			with open("data/submission.txt", "w", encoding="utf8") as f:
				f.write("\n".join(reported_datas))

			return None, None

		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

		return EER, minDCF
	def infer(self, eval_list :str, save_path ):
		self.eval()
		files = []
		embeddings = {}
		# lines = open(eval_list).read().splitlines()
		lines = [line.split() for line in open(eval_list).read().split("\n") if line]
		for line in lines:
			files.append(line[0])
			files.append(line[1])
		setfiles = list(set(files))
		setfiles.sort()

		for file in tqdm.tqdm(setfiles, total=len(setfiles)):
			try:
				data_1, data_2 = self._extract_feature(file)
				
				# speaker embeddings
				with torch.no_grad():
					embedding_1 = self._forward(data_1, aug=False)
					embedding_1 = F.normalize(embedding_1, p=2, dim=1).mean(dim=0, keepdim=True)
					embedding_2 = self._forward(data_2, aug=False)
					embedding_2 = F.normalize(embedding_2, p=2, dim=1).mean(dim=0, keepdim=True)

				embeddings[file] = [embedding_1, embedding_2]
			except:
				continue
		scores, labels  = [], []
		count_fail = 0
		failed_lines = []
		for line in lines:
			try:
				embedding_11, embedding_12 = embeddings[line[0]]
				embedding_21, embedding_22 = embeddings[line[1]]
				
				# compute the scores
				score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
				score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
				score = (score_1 + score_2) / 2
				score = score.detach().cpu().numpy()
				scores.append(score)
			except:
				count_fail += 1
				failed_lines.append(line)
				continue
		with open(save_path, "w") as f:
			for score in scores:
   				f.write(str(score)+"\n")
				
		print("number of failed audios: ", count_fail)
		with open(save_path.replace(".txt", "_failed.txt"), "w") as f:
			for line in failed_lines:
				f.write("\t".join(line) + "\n")
		return 0
   
	def eval_network_with_asnorm(self, eval_list: str, cohort_embs: np.array):
		self.eval()
		files = []
		embeddings = {}
		lines = [line.split() for line in open(eval_list).read().split("\n") if line]
		for line in lines:
			files.append(line[0])
			files.append(line[1])
		setfiles = list(set(files))
		setfiles.sort()

		for _, file in tqdm.tqdm(enumerate(setfiles), total=len(setfiles)):
			data_1, data_2 = self._extract_feature(file)	
			# speaker embeddings
			with torch.no_grad():
				embedding_1 = self._forward(data_1, aug=False)
				embedding_1 = F.normalize(embedding_1, p=2, dim=1)
				embedding_2 = self._forward(data_2, aug=False)
				embedding_2 = F.normalize(embedding_2, p=2, dim=1).mean(dim=0, keepdim=True)
			embeddings[file] = [embedding_1, embedding_2]
		
		scores, labels  = [], []
		for line in tqdm.tqdm(lines, desc="scoring"):
			embedding_11, embedding_12 = embeddings[line[0]]
			embedding_21, embedding_22 = embeddings[line[1]]
			# compute the scores
			score_1 = as_norm(
				torch.mean(torch.matmul(embedding_11, embedding_21.T)), 
				embedding_11, 
				embedding_21, 
				cohort_embs[0]
			).detach().cpu().numpy()

			score_2 = as_norm(
				torch.mean(torch.matmul(embedding_12, embedding_22.T)), 
				embedding_12, 
				embedding_22, 
				cohort_embs[1]
			).detach().cpu().numpy()

			score = (score_1 + score_2) / 2
			scores.append(score)
			labels.append(int(line[2]))

		# Coumpute EER and minDCF
		EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
		fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
		minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

		return EER, minDCF
	
	def build_cohort(self, train_list: DataLoader):
		self.eval()
		embeddings = [[], []]
		for filename in tqdm.tqdm(train_list, total = len(train_list)):
			data_1, data_2 = self._extract_feature(filename)
			
			# speaker embeddings
			with torch.no_grad():
				embedding_1 = self._forward(data_1, aug=False)
				embeddings[0].append(F.normalize(embedding_1, p=2, dim=1))
				embedding_2 = self._forward(data_2, aug=False)
				embeddings[1].append(F.normalize(embedding_2, p=2, dim=1).mean(dim=0, keepdim=True))
			
		return [torch.stack(x) for x in embeddings]

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


def as_norm(score, embedding_1, embedding_2, cohort_feats, topk=1000):
	score_1 = torch.matmul(cohort_feats, embedding_1.T)[:,0]
	score_1 = torch.topk(score_1, topk, dim = 0)[0]
	mean_1  = torch.mean(score_1, dim = 0)
	std_1   = torch.std(score_1, dim = 0)

	score_2 = torch.matmul(cohort_feats, embedding_2.T)[:,0]
	score_2 = torch.topk(score_2, topk, dim = 0)[0]
	mean_2  = torch.mean(score_2, dim = 0)
	std_2   = torch.std(score_2, dim = 0)

	score = 0.5 * (score - mean_1) / std_1 + 0.5 * (score - mean_2) / std_2
	return score

import os
import json
import glob
import torch
import random
import librosa
import soundfile
import numpy as np
from scipy import signal


class data_augmentation(object):
	def __init__(self, musan_path: str, rir_path: str) -> None:
		
		self.sample_rate = 16000
		self.max_length  = 200 * 240 + 160
		self.noisetypes  = ["background","speech","music"]
		self.noisesnr	= {"background":[0, 15],"speech":[13, 20],"music":[5, 15]}
		self.numnoise	= {"background":[1, 1], "speech":[3, 8], "music":[1, 1]}

		self.noiselist   = {_noise: glob.glob(os.path.join(musan_path, _noise, "*/*.wav")) for _noise in self.noisetypes}
		print("[*] Total noise sample: ")
		for k, v in self.noiselist.items():
			print(f"- {k}: {len(v)} samples")
		self.rir_files   = glob.glob(os.path.join(rir_path, "*/*/*.wav"))
		print(f"- reverse: {len(self.rir_files)} samples")
		
	def add_rev(self, audio: np.array):
		rir_file = random.choice(self.rir_files)
		rir, sr  = librosa.load(rir_file, sr=self.sample_rate)
		rir 	 = np.expand_dims(rir.astype(np.float),0)
		rir 	 = rir / np.sqrt(np.sum(rir**2))
		return signal.convolve(audio, rir, mode="full")[:,:self.max_length]

	def add_noise(self, audio: np.array, noisecat: str):
		clean_db  = 10 * np.log10(np.mean(audio ** 2) + 1e-4) 
		numnoise  = self.numnoise[noisecat]
		noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
		noises 	  = []
		for noise in noiselist:
			noiseaudio, sr = soundfile.read(noise)
			if noiseaudio.shape[0] <= self.max_length:
				shortage = self.max_length - noiseaudio.shape[0]
				noiseaudio = np.pad(noiseaudio, (0, shortage), "wrap")
			start_frame = np.int64(random.random()*(noiseaudio.shape[0] - self.max_length))
			noiseaudio  = noiseaudio[start_frame:start_frame + self.max_length]
			noiseaudio  = np.stack([noiseaudio],axis=0)
			noise_db	= 10 * np.log10(np.mean(noiseaudio ** 2)+1e-4) 
			noisesnr	= random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(np.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = np.sum(np.concatenate(noises,axis=0), axis=0, keepdims=True)
		
		return noise + audio


class train_loader(object):
	def __init__(self, train_list: str, musan_path: str, rir_path: str, sample_rate: int, max_length: str, augment: bool):
		
		# Load and configure augmentation files
		self.sample_rate = sample_rate
		self.max_length  = max_length
		self.augment_module = None
		if augment is True:
			self.augment_module = data_augmentation(musan_path, rir_path)
			self.augment_module.sample_rate = self.sample_rate
			self.augment_module.max_length  = self.max_length

		# Load data & labels
		self.data_list  = []
		self.data_label = []
		lines = open(train_list).read().splitlines()
		dictkeys = list(set([x.split("|")[1] for x in lines]))
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys)}
		for index, line in enumerate(lines):
			file_name, speaker_label = line.split("|")
			speaker_label = dictkeys[speaker_label]
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)

	def __getitem__(self, index):
		# Read the utterance and randomly select the segment
		# audio, sr = librosa.load(self.data_list[index], sr=self.sample_rate)
		audio, sr = soundfile.read(self.data_list[index])
		# audio = audio[0]
		if audio.ndim == 2:
			audio = audio[:, 0]
		if audio.shape[0] <= self.max_length:
			shortage = self.max_length - audio.shape[0]
			audio = np.pad(audio, (0, shortage), "wrap")
		start_frame = np.int64(random.random() * (audio.shape[0] - self.max_length))
		audio = audio[start_frame: start_frame + self.max_length]
		audio = np.stack([audio], axis=0)
		
		# Data Augmentation
		augtype = random.randint(0, 4) if self.augment_module is not None else 0
		if augtype == 0:   # Original
			audio = audio
		elif augtype == 1: # Reverberation
			audio = self.augment_module.add_rev(audio)
		elif augtype == 2: # Babble
			audio = self.augment_module.add_noise(audio, "speech")
		elif augtype == 3: # Music
			audio = self.augment_module.add_noise(audio, "music")
		elif augtype == 4: # Noise
			audio = self.augment_module.add_noise(audio, "background")
		elif augtype == 5: # Television noise
			audio = self.augment_module.add_noise(audio, "speech")
			audio = self.augment_module.add_noise(audio, "music")
		
		return torch.FloatTensor(audio[0]), self.data_label[index]

	def __len__(self):
		return len(self.data_list)


class verify_loader(object):
	def __init__(self, train_list: str, musan_path: str, rir_path: str, sample_rate: int, max_length: str, augment: bool):
		
		# Load and configure augmentation files
		self.sample_rate = sample_rate
		self.max_length  = max_length
		self.augment_module = None
		if augment is True:
			self.augment_module = data_augmentation(musan_path, rir_path)
			self.augment_module.sample_rate = self.sample_rate
			self.augment_module.max_length  = self.max_length

		# Load data & labels
		self.datalist: dict = json.load(open(train_list, "r", encoding="utf8"))

	def load_audio(self, fpath: str) -> np.array:
		# Read the utterance and randomly select the segment
		audio, sr= soundfile.read(fpath)
		if audio.shape[0] <= self.max_length:
			shortage = self.max_length - audio.shape[0]
			audio = np.pad(audio, (0, shortage), "wrap")
		start_frame = np.int64(random.random() * (audio.shape[0] - self.max_length))
		audio = audio[start_frame: start_frame + self.max_length]
		audio = np.stack([audio], axis=0)
		
		# Data Augmentation
		augtype = random.randint(0, 4) if self.augment_module is not None else 0
		if augtype == 0:   # Original
			audio = audio
		elif augtype == 1: # Reverberation
			audio = self.augment_module.add_rev(audio)
		elif augtype == 2: # Babble
			audio = self.augment_module.add_noise(audio, "speech")
		elif augtype == 3: # Music
			audio = self.augment_module.add_noise(audio, "music")
		elif augtype == 4: # Noise
			audio = self.augment_module.add_noise(audio, "background")
		elif augtype == 5: # Television noise
			audio = self.augment_module.add_noise(audio, "speech")
			audio = self.augment_module.add_noise(audio, "music")

		return audio

	def generate_one_samples(self):
		ans_type = random.randint(0, 1)
		if ans_type == 1:  # target
			spk = random.choice(list(self.datalist.keys()))
			enr, tst = random.choices(self.datalist[spk]["bonafide"], k=2)
		
		elif ans_type == 0:  # nontarget
			nontarget_type = random.randint(1, 2)

			if nontarget_type == 1:  # zero-effort nontarget
				spk, ze_spk = random.choices(list(self.datalist.keys()), k=2)
				enr = random.choice(self.datalist[spk]["bonafide"])
				tst = random.choice(self.datalist[ze_spk]["bonafide"])

			if nontarget_type == 2:  # spoof nontarget
				spk = random.choice(list(self.datalist.keys()))
				if len(self.datalist[spk]["spoof"]) == 0:
					while True:
						spk = random.choice(list(self.datalist.keys()))
						if len(self.datalist[spk]["spoof"]) != 0:
							break
				enr = random.choice(self.datalist[spk]["bonafide"])
				tst = random.choice(self.datalist[spk]["spoof"])
		
		wav_asv_enr = self.load_audio(enr)[0]
		wav_asv_tst = self.load_audio(tst)[0]

		return torch.FloatTensor(wav_asv_enr), torch.FloatTensor(wav_asv_tst), ans_type

	def generate_inputs(self, batch_size: int):
		inputs = [self.generate_one_samples() for i in range(batch_size)]

		enr_datas = torch.stack([x[0] for x in inputs], dim=0)
		tst_datas = torch.stack([x[1] for x in inputs], dim=0)
		ans_types = torch.LongTensor([x[2] for x in inputs])

		return (enr_datas, tst_datas, ans_types)

	def __len__(self):
		# NOTE(ducnt2): consider __len__ is repeat iters
		return sum([len(v) for v in self.datalist.values()])

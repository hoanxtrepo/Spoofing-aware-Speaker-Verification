"""
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
"""

import os
import sys
import glob
import time
import tqdm
import math
import yaml

import soundfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import warnings
from models.asv_cm import ASVCMClassifier
from src.tools.utils import *
from src.tools.dataloader import verify_loader


def do_train():
	global config
	## Define the data loader
	config = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
	trainloader = verify_loader(
		train_list  = args.train_list,
		musan_path  = args.musan_path,
		rir_path    = args.rir_path,
		sample_rate = config["preprocessing"]["signal"]["sampling_rate"],
		max_length  = config["preprocessing"]["num_frames"] * 160 + 240,
		augment		= False
	)
	evalLoader = [x.split("\t") for x in open(args.eval_list).read().split("\n") if x]
	files = []
	for line in evalLoader:
		files.extend(line[1:])
	setfiles = list(set(files))
	setfiles.sort()

	## Search for the exist models
	modelfiles = glob.glob("%s/model_0*.model"%args.model_save_path)
	modelfiles.sort()

	epoch = 1
	lr = 0.0001
	## If asv_checkpoint and cm_checkpoint, system will train from the asv_checkpoint and cm_checkpoint
	if args.asv_checkpoint and args.cm_checkpoint: 
		print(f"Model ASV loaded from  {args.asv_checkpoint}")
		print(f"Model CM loaded from  {args.cm_checkpoint}")


		nnet = ASVCMClassifier(
			asv_hparams = yaml.load(open(f"{os.path.dirname(args.asv_checkpoint)}/config.yaml", "r"), Loader=yaml.FullLoader), 
			cm_hparams  = yaml.load(open(f"{os.path.dirname(args.cm_checkpoint)}/config.yaml", "r"), Loader=yaml.FullLoader),
		).to(device)
		nnet.asv_emb.load_parameters(args.asv_checkpoint)
		nnet.cm_emb.load_parameters(args.cm_checkpoint)

		# NOTE: saving model config
		config = {"asv_hparams": nnet.asv_hparams, "cm_hparams": nnet.cm_hparams}
		with open(os.path.join(args.model_save_path, "config.yaml"), "w", encoding="utf8") as f:
			yaml.dump(config, f, default_flow_style=False)
	elif args.initial_model:
		print(f"Model loaded from  {args.initial_model}")
		config = yaml.load(open(f"{os.path.dirname(args.initial_model)}/config.yaml", "r"), Loader=yaml.FullLoader)
		nnet = ASVCMClassifier(**config).to(device)
		nnet.load_parameters(args.initial_model)
		
	## Otherwise, system will try to start from the saved model&epoch
	elif len(modelfiles) >= 1:
		print(f"Model loaded from  {modelfiles[-1]}")
		config = yaml.load(open(f"{os.path.dirname(modelfiles[-1])}/config.yaml", "r"), Loader=yaml.FullLoader)
		nnet = ASVCMClassifier(**config).to(device)
		nnet.load_parameters(modelfiles[-1])
		
		epoch = int(os.path.basename(modelfiles[-1]).replace(".model", "").split("_")[1]) + 1

	## Otherwise, system will train from scratch
	else:
		raise NotImplementedError(f"This training must be trained by 2 new {args.asv_checkpoint} and {args.cm_checkpoint} or resuming training from {args.model_save_path}")

	## Initialize loss & optimize 
	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.Adam(nnet.parameters(), lr=lr, weight_decay=0.0001)
	scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

	results    = []
	score_file = open(args.score_save_path, "a+")
	while(1):
		## Training for one epoch
		nnet.train()
		num, index, top1, loss = 1, 0, 0, 0
		lr = optimizer.param_groups[0]["lr"]
		while num < args.iter_per_epoch:
			enr_datas, tst_datas, labels = trainloader.generate_inputs(batch_size=args.batch_size)
			labels = labels.float().to(device)
			feats, preds = nnet(asv_enr_wav = enr_datas.to(device), asv_tst_wav = tst_datas.to(device))
			nloss = criterion(preds, labels)

			optimizer.zero_grad()
			nloss.backward()
			optimizer.step()
			
			index += len(labels)
			top1  += (torch.sigmoid(preds).round() == (labels)).float().mean() * 100
			loss  += nloss.detach().cpu().numpy()
			sys.stderr.write(time.strftime("%m-%d %H:%M:%S") + \
			" [%2d] Lr: %5f, Training: %.2f%%, "	%(epoch, lr, 100 * (num / args.iter_per_epoch)) + \
			" Loss: %.5f, ACC: %2.2f%% \r"		%(loss/(num), top1/index*len(labels)))
			sys.stderr.flush()
			num += 1

		trn_acc = top1/index*len(labels)
		scheduler.step()
		sys.stdout.write("\n")

		## Evaluation every [test_step] epochs
		if epoch % args.test_step == 0:
			nnet.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
			nnet.eval()
			embeddings = {}
			for file in tqdm.tqdm(setfiles):
				data, _ = soundfile.read(file)
				data = torch.FloatTensor(pad(data, args.eval_max_length)).unsqueeze(0).to(device)
				with torch.no_grad():
					embeddings[file] = {
						"asv": nnet.asv_emb._forward(data), 
						"cm": nnet.cm_emb._forward(data)
					}

			# results mean EERs
			scores, labels  = [], []
			for line in evalLoader:
				with torch.no_grad():
					_, preds = nnet.validate(
						asv_enr_emb = embeddings[line[1]]["asv"],
						asv_tst_emb = embeddings[line[2]]["asv"],
						spf_emb = embeddings[line[1]]["cm"]
					)
				scores.append(preds.item())
				if len(labels) > 2: labels.append(int(line[0]))

			# Coumpute EER and minDCF
			if len(labels):
				EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
				results.append(EER)
				print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, trn_acc, results[-1], min(results)))
				score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, trn_acc, results[-1], min(results)))
				score_file.flush()
			else:
				reported_datas = []
				for i in range(len(scores)):
					reported_datas.append(f"{'/'.join(evalLoader[i][0].split('/')[-2: ])}\t{'/'.join(evalLoader[i][1].split('/')[-2: ])}\t{scores[i]}")
				# with open(f"data/submission.model_{epoch}.txt", "w", encoding="utf8") as f:
				# 	f.write("\n".join(reported_datas))
				score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%\n"%(epoch, lr, loss, trn_acc))
				score_file.flush()

		
		if epoch >= args.max_epoch: 
			quit()

		epoch += 1


def do_evaluation():
	config = yaml.load(open(f"{os.path.dirname(args.initial_model)}/config.yaml", "r"), Loader=yaml.FullLoader)
	nnet = ASVCMClassifier(**config).to(device)
	print("Model loaded from %s "%args.initial_model)
	nnet.load_parameters(args.initial_model)
	nnet.eval()
	
	# evaling
	evalLoader = [x.split() for x in open(args.eval_list).read().split("\n") if x]
	files = []
	for line in evalLoader:
		files.extend(line[: 2])
	setfiles = list(set(files))
	setfiles.sort()

	embeddings = {}
	for file in tqdm.tqdm(setfiles, desc="Build embeddings"):
		audio, _ = soundfile.read(file)
		data = torch.FloatTensor(pad(audio, args.eval_max_length)).unsqueeze(0).to(device)

		with torch.no_grad():
			embeddings[file] = {"asv": nnet.asv_emb._forward(data), "cm": nnet.cm_emb._forward(data)}

	# results mean EERs
	scores, labels  = [], []
	for line in tqdm.tqdm(evalLoader, desc="Calculate score"):
		with torch.no_grad():
			_, preds = nnet.validate(
				asv_enr_emb = embeddings[line[0]]["asv"],
				asv_tst_emb = embeddings[line[1]]["asv"],
				spf_emb = embeddings[line[0]]["cm"]
			)
		scores.append(preds.item())
		if len(line) == 3:
			labels.append(int(line[2]))
	bestEER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
	fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
	minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
	print("EER %2.2f%% minDCF %2.2f%%"%(bestEER, minDCF))
	# if len(labels):
	# 	# Coumpute EER and minDCF
	# 	bestEER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
	# 	fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
	# 	minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

	# 	print("EER %2.2f%% minDCF %2.2f%%"%(bestEER, minDCF))
	# else:
	# 	reported_datas = []
	# 	for i in range(len(scores)):
	# 		reported_datas.append(f"{'/'.join(evalLoader[i][0].split('/')[-2: ])}\t{'/'.join(evalLoader[i][1].split('/')[-2: ])}\t{scores[i]}")
	# 	with open(f"data/submission.{os.path.basename(args.initial_model)}.txt", "w", encoding="utf8") as f:
	# 		f.write("\n".join(reported_datas))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "ECAPA_trainer")
	## Training and evaluation path/lists, save path
	parser.add_argument("--train_list", type=str,   	default="/home4/vuhl/hoanxt/vsasv-vlsp-code/data/data_english/joint_data/train2.json",
						help="The path of the training list")
	parser.add_argument("--eval_list",  type=str,   	default="/home4/vuhl/hoanxt/vsasv-vlsp-code/data/joint_data/only_adversarial.txt",
						help="The path of the evaluation list")
	parser.add_argument("--cohort_path", type=str, 		default=None,
						help="The path of the evaluation data")
	parser.add_argument("--musan_path", type=str,   	default="",
						help="The path to the MUSAN set")
	parser.add_argument("--rir_path",   type=str,   	default="",
						help="The path to the RIR set")
	parser.add_argument("--save_path",  type=str,   	default="exps/joint_voxceleb",
						help="Path to save the score.txt and models")
	
	## Modelling Settings
	parser.add_argument("--asv_checkpoint", type=str,	default="/home4/vuhl/hoanxt/vsasv-vlsp-code/exps/ASV_voxceleb/model/model_0100.model",
		     			help="Direct path to ASV checkpoint")
	parser.add_argument("--cm_checkpoint", 	type=str,	default="/home4/vuhl/hoanxt/vsasv-vlsp-code/exps/CM_voxceleb/model/model_0010.model",
		     			help="Direct path to CM checkpoint")
	parser.add_argument("--initial_model",  type=str,	default="/home4/vuhl/hoanxt/vsasv-vlsp-code/exps/joint_clean_with_se/model/model_0032.model",
						help="Path of the initial_model")

	## Training Settings
	parser.add_argument("--max_epoch",  type=int,   default=20,    help="Maximum number of epochs")
	parser.add_argument("--batch_size", type=int,   default=32,		help="Batch size")
	parser.add_argument("--n_cpu",      type=int,   default=4,      help="Number of loader threads")
	parser.add_argument("--iter_per_epoch", type=int, default=2000, help="Repeat iteration per epochs")
	parser.add_argument("--test_step",  type=int,   default=10,      help="Test and save every [test_step] epochs")

	## Command
	parser.add_argument("--eval",    dest="eval", 	 action="store_true", help="Only do evaluation")
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	## Initialization
	warnings.simplefilter("ignore")
	torch.multiprocessing.set_sharing_strategy("file_system")
	args = parser.parse_args()
	args = init_args(args)
	
	args.eval_max_length = 1000 * 240 + 160 
	## Initial frontend & backend
	if args.eval:
		print("evaluating")
		if os.path.isdir(args.initial_model):
			import glob
			modelfiles = sorted(glob.glob("%s/model_*.model"%args.initial_model))
			## NOTE: skip this when real play
			# modelfiles = [fpath for fpath in modelfiles if int(os.path.basename(fpath).split(".")[0].replace("model_", "")) > 210]
			for fpath in modelfiles[-1: ]:
				args.initial_model = fpath
				do_evaluation()
		else:
			do_evaluation()
	else:
		if args.initial_model and os.path.isdir(args.initial_model): 
			raise NotImplementedError("Initial model when training must be file path")
		do_train()
#/home4/vuhl/hoanxt/vsasv-vlsp-code/exps/ASV_clean/model/config.yaml
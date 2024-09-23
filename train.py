"""
This is the main code of the ECAPATDNN project, to define the parameters and build the construction
"""

import os
import glob
import time
import json
import yaml
import torch
import argparse
import warnings
import numpy as np
from torch.utils.data import DataLoader
from src.tools.utils import *
from src.tools.dataloader import train_loader


def do_train():
	global config
	## Define the data loader
	trainloader = train_loader(
		train_list  = args.train_list,
		musan_path  = args.musan_path,
		rir_path    = args.rir_path,
		sample_rate = config["preprocessing"]["signal"]["sampling_rate"],
		max_length  = config["preprocessing"]["num_frames"] * 160 + 240,
		augment		= False
	)
	trainLoader = DataLoader(
		trainloader, 
		batch_size  = args.batch_size, 
		shuffle     = True, 
		num_workers = args.n_cpu, 
		drop_last   = True
	)

	## Search for the exist models
	modelfiles = glob.glob("%s/model_0*.model"%args.model_save_path)
	modelfiles.sort()

	## If initial_model is exist, system will train from the initial_model
	if args.initial_model:
		print("Model %s loaded from "%args.initial_model)
		s = Classifier(
			n_class  = classes, 
			hparams  = config,
			frontend_type = args.frontend_type,
			backend_type  = args.backend_type
		)
		s.load_parameters(args.initial_model)
		epoch = 1

	## Otherwise, system will try to start from the saved model&epoch
	elif len(modelfiles) >= 1:
		print("Model %s loaded from "%modelfiles[-1])
		epoch = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][6:]) + 1
		s = Classifier(
			n_class  = classes, 
			hparams  = config,
			frontend_type = args.frontend_type,
			backend_type  = args.backend_type
		)
		s.load_parameters(modelfiles[-1])

	## Otherwise, system will train from scratch
	else:
		epoch = 1
		s = Classifier(
			n_class  = classes, 
			hparams  = config,
			frontend_type = args.frontend_type,
			backend_type  = args.backend_type
		)
	
	results = []
	config = {
		"model": {
			"frontend": args.frontend_type,
			"backend": args.backend_type
		},
		"hparams": {
			**{k: v for k, v in config.items() if k in ["preprocessing", "loss_func", "loss", "optim", args.frontend_type, args.backend_type]}
		}
	}
	with open(os.path.join(args.model_save_path, "config.yaml"), "w", encoding="utf8") as f:
		yaml.dump(config, f, default_flow_style=False)
	score_file = open(args.score_save_path, "a+")
	while(1):		
		## Training for one epoch
		loss, lr, acc = s.train_network(epoch=epoch, loader=trainLoader)

		## Evaluation every [test_step] epochs
		if epoch % args.test_step == 0:

			s.save_parameters(args.model_save_path + "/model_%04d.model"%epoch)
			print("checkpoint saved at ",args.model_save_path + "/model_%04d.model"%epoch)
			# results mean EER
			results.append(s.eval_network(eval_list=args.eval_list)[0])
			if args.model == "asv":
				print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%"%(epoch, acc, results[-1], min(results)))
				score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, EER %2.2f%%, bestEER %2.2f%%\n"%(epoch, lr, loss, acc, results[-1], min(results)))
			else:
				print(time.strftime("%Y-%m-%d %H:%M:%S"), "%d epoch, ACC %2.2f%%, F1 %2.2f%%, bestF1 %2.2f%%"%(epoch, acc, results[-1], max(results)))
				score_file.write("%d epoch, LR %f, LOSS %f, ACC %2.2f%%, F1 %2.2f%%, bestF1 %2.2f%%\n"%(epoch, lr, loss, acc, results[-1], max(results)))
			score_file.flush()

		if epoch >= args.max_epoch: 
			quit()

		epoch += 1


def do_evaluation():
	global config
	## Only do evaluation, the initial_model is necessary
	if args.eval == True:
		s = Classifier(
			n_class  = classes, 
			hparams  = config["hparams"],
			frontend_type = config["model"]["frontend"],
			backend_type  = config["model"]["backend"]
		)

		print("Model %s loaded from "%args.initial_model)
		s.load_parameters(args.initial_model)

		if args.cohort_path is not None:
			if os.path.exists(args.cohort_path):
				cohort_feats = [torch.from_numpy(x).to(device) for x in np.load(args.cohort_path)]
			else:
				trainloader = train_loader(**vars(args))
				cohort_feats = s.build_cohort(trainloader.data_list)
				np.save(args.cohort_path, [x.detach().cpu().numpy() for x in cohort_feats])

			bestEER, minDCF = s.eval_network_with_asnorm(eval_list = args.eval_list, cohort_embs = cohort_feats)
		else:
			# bestEER, minDCF = s.eval_network(eval_list = args.eval_list)
			s.infer(eval_list = args.eval_list, save_path = args.score_path)
		#print("EER %2.2f%%, minDCF %.4f%%"%(bestEER, minDCF))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = "ECAPA_trainer")
	## Training and evaluation path/lists, save path
	parser.add_argument("--train_list", type=str,   	default="/home4/vuhl/hoanxt/vsasv-vlsp-code/data/cm_data/train_with_TTS.txt",
						help="The path of the training list")
	parser.add_argument("--eval_list",  type=str,   	default="/home4/vuhl/hoanxt/vsasv-vlsp-code/data/joint_data/only_adversarial.txt",
						help="The path of the evaluation list")
	parser.add_argument("--score_path",  type=str,   	default="/home4/vuhl/hoanxt/vsasv-vlsp-code/exps/ASV_clean/only_adversarial.txt",
						help="The path of the infer score")
	parser.add_argument("--cohort_path", type=str, 		default=None,
						help="The path of the evaluation data")
	parser.add_argument("--musan_path", type=str,   	default="",
						help="The path to the MUSAN set")
	parser.add_argument("--rir_path",   type=str,   	default="",
						help="The path to the RIR set")
	parser.add_argument("--save_path",  type=str,   	default="",
						help="Path to save the score.txt and models")
	parser.add_argument("--initial_model",  type=str,	default="/home4/vuhl/hoanxt/vsasv-vlsp-code/exps/ASV_clean/model/model_0100.model",
						help="Path of the initial_model")
 # Ecapa checkpoint: /home4/vuhl/hoanxt/vsasv-vlsp-code/exps/ASV_clean/model/model_0100.model
 #aasit checkpont: /home4/vuhl/hoanxt/vsasv-vlsp-code/exps/CM/model/model_0006.model
	## Modelling Settings
	parser.add_argument("--model", 			type=str, 	default="asv",		choices=["asv", "cm"],
					 	help="Choice task training model task")
	parser.add_argument("--frontend_type", 	type=str,	default="none", 	choices=["mel_spectrogram", "sinc_net", "none"],
		     			help="Choice front-end for main model")
	parser.add_argument("--backend_type", 	type=str,	default="assist",	choices=["assist", "ecapa_tdnn", "ds_tdnn", "xvector"],
		     			help="Choice back-end for main model")
	## Training Settings
	parser.add_argument("--max_epoch",  type=int,   default=6,    help="Maximum number of epochs")
	parser.add_argument("--batch_size", type=int,   default=32,		help="Batch size")
	parser.add_argument("--n_cpu",      type=int,   default=4,      help="Number of loader threads")
	parser.add_argument("--test_step",  type=int,   default=5,      help="Test and save every [test_step] epochs")

	## Command
	parser.add_argument("--eval",    dest="eval", 	 action="store_true", help="Only do evaluation")

	## Initialization
	warnings.simplefilter("ignore")
	torch.multiprocessing.set_sharing_strategy("file_system")
	args = parser.parse_args()
	args = init_args(args)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	
	## Initialize model classifier
	if args.model == "asv":
		from models.asv import ASVClassifier as Classifier
	else:
		from models.cm import CMClassifier as Classifier

	## Initial frontend & backend
	if args.eval:
		config	 = yaml.load(open(f"{os.path.dirname(args.initial_model)}/config.yaml", "r"), Loader=yaml.FullLoader)
		speakers = None
		classes  = 1
		do_evaluation()
	else:
		config 	 = yaml.load(open("config/config.yaml", "r"), Loader=yaml.FullLoader)
		# speakers = json.load(open(os.path.join(os.path.dirname(args.train_list), "speakers.json"), "r"))
		speakers = json.load(open("/home4/vuhl/hoanxt/vsasv-vlsp-code/data/cm_data/speakers.json", "r"))
		classes  = len(speakers) if args.model == "asv" else 1
		do_train()

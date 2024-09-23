#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import accuracy


class NNNLoss(nn.Module):
	def __init__(self, nOut, nClasses, **kwargs):
		super(NNNLoss, self).__init__()

		self.fc		   = nn.Linear(nOut, nClasses)
		self.softmax   = nn.LogSoftmax(dim=1)
		self.criterion = nn.NLLLoss()

		print("Initialised The negative log Likelihood Loss")

	def forward(self, x: torch.Tensor, label: torch.Tensor=None):

		x 	  = self.fc(x)
		nloss = self.criterion(self.softmax(x), label)
		prec1 = accuracy(x.detach(), label.detach(), topk=(1,))[0]

		return nloss, prec1

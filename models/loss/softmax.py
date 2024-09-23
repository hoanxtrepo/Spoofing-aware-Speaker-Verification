#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from .utils import accuracy


class Softmax(nn.Module):
	def __init__(self, nOut, nClasses, **kwargs):
		super(Softmax, self).__init__()

		self.test_normalize = True
		
		self.fc 		= nn.Linear(nOut, nClasses)
		self.criterion  = torch.nn.CrossEntropyLoss()

		print('Initialised Softmax Loss')

	def forward(self, x: torch.Tensor, label: torch.Tensor=None):

		x 	  = self.fc(x)
		nloss = self.criterion(x, label)
		prec1 = accuracy(x.detach(), label.detach(), topk=(1,))[0]

		return nloss, prec1
import os
import sys
import random
import warnings

import numpy as np
import pandas as pd
import time

import matplotlib.pyplot as plt
import io
import pickle
import json

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn.functional as F

from uNetModel import *
from encoder import *

# For this cell used same code from PyTorch notebook in assignment 2 of Stanford's CS231n Spring 2018 offering
USE_GPU = True
dtype = torch.float32
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    dtype = torch.float32
seed = 1

# This piece of code was borrowed from the Imagenet [7] github
def unpickle(file):
	with open(file, 'rb') as fo:
		dict = pickle.load(fo)
	return dict

def calculatePerformance(X_perf, Y_perf, model, mini_batch_size =  64):
	model.eval() #set model to evaluation mode
	numCorrect = 0
	num_batches = int(X_perf.shape[0]/mini_batch_size)
	num_remaining = X_perf.shape[0] - num_batches *mini_batch_size
	with torch.no_grad():
		for i in range(num_batches):
			x = torch.from_numpy(X_perf[i*mini_batch_size:(i+1)*mini_batch_size, :, :, :])
			x = x.to(device = device, dtype = dtype)
			preds = model(x).cpu().numpy()
			preds = np.argmax(preds, axis = 1) + 1
			numCorrect += np.sum(preds == Y_perf[i*mini_batch_size:(i+1)*mini_batch_size])
		x = torch.from_numpy(X_perf[num_batches*mini_batch_size:, :, :, :])
		x = x.to(device = device, dtype = dtype)
		preds = model(x).cpu().numpy()
		preds = np.argmax(preds, axis = 1) + 1
		numCorrect += np.sum(preds == Y_perf[num_batches*mini_batch_size:])
	return numCorrect/(int(X_perf.shape[0]))

def trainEncoder(model, x_train, y_train, optimizer, epochs = 1, mini_batch_size = 64, noVal = False):
	model = model.to(device=device)  # move the model parameters to CPU/GPU
	T = 0
	num_batches = int(x_train.shape[0]/mini_batch_size)
	num_remaining = x_train.shape[0] - num_batches *mini_batch_size
	loss_history = []
	for e in range(epochs):
		for t in range(num_batches):
			rand_indices = np.random.choice(x_train.shape[0], mini_batch_size)
			x = torch.from_numpy(x_train[rand_indices, :, :, :])
			y = torch.from_numpy(y_train[rand_indices] - 1) # cross_entropy fn expects 0<= y[i] <=C-1
			model.train()  # put model to training mode
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=dtype)

			preds = model(x)
			loss = F.cross_entropy(preds, y.type(torch.long))

			# Zero out all of the gradients for the variables which the optimizer
			# will update.
			optimizer.zero_grad()

			# This is the backwards pass: compute the gradient of the loss with
			# respect to each  parameter of the model.
			loss.backward()

			# Actually update the parameters of the model using the gradients
			# computed by the backwards pass.
			optimizer.step()

			if T % print_every == 0:
				currLoss = loss.item()
				loss_history.append(currLoss)
				print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, currLoss))
			if (num_remaining <= 0 and t == (num_batches -1)):
				percCorrect = calculatePerformance(x_train, y_train, model)
				print('Train percentage at epoch %d is %.4f' % (e, percCorrect))
				if (noVal == False):
					percCorrect = calculatePerformance(x_val, y_val, model)
					print('Train percentage at epoch %d is %.4f' % (e, percCorrect))
			T +=1
		if num_remaining > 0:
			rand_indices = np.random.choice(len(x_train), num_remaining)
			x = torch.from_numpy(x_train[rand_indices, :, :, :])
			y = torch.from_numpy(y_train[rand_indices] -1)
			model.train()  # put model to training mode
			x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
			y = y.to(device=device, dtype=dtype)

			preds = model(x)
			loss = F.cross_entropy(preds, y.type(torch.long))

			# Zero out all of the gradients for the variables which the optimizer
			# will update.
			optimizer.zero_grad()

			# This is the backwards pass: compute the gradient of the loss with
			# respect to each  parameter of the model.
			loss.backward()

			# Actually update the parameters of the model using the gradients
			# computed by the backwards pass.
			optimizer.step()
			if T % print_every == 0:
				currLoss = loss.item()
				loss_history.append(currLoss)
				print('Epoch %d, Iteration %d, loss = %.4f' % (e, num_batches, currLoss))
			percCorrect = calculatePerformance(x_train, y_train, model)
			print('Train percentage at epoch %d is %.4f' % (e, percCorrect))
			if (noVal == False):
				percCorrect = calculatePerformance(x_val, y_val, model)
				print('Val percentage at epoch %d is %.4f' % (e, percCorrect))
			T +=1
	return calculatePerformance(x_train, y_train, model),loss_history

# Constant to control how frequently we print train loss
print_every = 100
def main():
	print('using device:', device)
	# For this cell, code belongs to [1]. Minor changes made to accomodate to our use 
	# (Using PyTorch instead of Keras/tensorflow)
	warnings.filterwarnings('ignore', category=UserWarning, module='skimage')
	random.seed = seed
	np.random.seed = seed

	# Reading imagenet data
	imgnetX = []
	imgnetY = []
	for i in range(5):
		d = unpickle('train_data_batch_'+str(i+1))
		imgnetX.append(d['data'])
		imgnetY.append(d['labels'])

	# Reshaping imgnetY such that shape is (N,)
	imgnetY = np.array(imgnetY)
	imgnetY = imgnetY.reshape((imgnetY.shape[0]*imgnetY.shape[1],))
	print('Shape of imgnetY = ', str(imgnetY.shape))

	# Reshaping imgnetX from (numDataBatches, dataBatchSize, img_size*img_size*3)
	# to (N, 3, img_size, img_size)
	# Two lines borrowed from [7]
	imgnetX = np.array(imgnetX)
	imgnetX = imgnetX.reshape((imgnetX.shape[0]*imgnetX.shape[1], imgnetX.shape[2]))
	imgnetX = np.dstack((imgnetX[:, :4096], imgnetX[:, 4096:8192], imgnetX[:, 8192:]))
	imgnetX = imgnetX.reshape((imgnetX.shape[0], 64, 64, 3)).transpose(0, 3, 1, 2)

	print('Shape of imgnetX = ', str(imgnetX.shape))

	shuffled_indices = np.random.permutation(imgnetX.shape[0])

	train_indices = shuffled_indices[0:int(.9*(imgnetX.shape[0]))]
	val_indices = shuffled_indices[int(.9*(imgnetX.shape[0])): int(.95*(imgnetX.shape[0]))]
	test_indices = shuffled_indices[int(.95*(imgnetX.shape[0])):]
	imgXtrain = imgnetX[train_indices, :, :, :]
	imgYtrain = imgnetY[train_indices]

	imgXval = imgnetX[val_indices, :, :, :]
	imgYval = imgnetY[val_indices]

	imgXtest = imgnetX[test_indices, :, :, :]
	imgYtest = imgnetY[test_indices]

	lrUsed = 0.00039293643571672977

# 	x_train = imgnetX[:500,:,:,:]
# 	y_train = imgnetY[:500]
# 	model = encoderNet()
# 	optimizer = optim.Adam(model.parameters(), lr = lrUsed)
# 	modelPerf, lossHistory = trainEncoder(model, x_train, y_train, optimizer, epochs = 100, noVal = True)

	print('Best lr used is ', str(lrUsed))
	imgNetModel = encoderNet()
	print_every = 100
	optimizer = optim.Adam(imgNetModel.parameters(), lr = lrUsed)
	modelPerf, lossHistory = trainEncoder(imgNetModel, imgXtrain, imgYtrain, optimizer, epochs = 10, noVal = True)
	torch.save(imgNetModel, 'imgNetModel')

if __name__ == '__main__':
	main()



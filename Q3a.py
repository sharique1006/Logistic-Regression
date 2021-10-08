import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math
import sys
import os

data_dir = sys.argv[1]
out_dir = sys.argv[2]
dataX = os.path.join(sys.argv[1], 'logisticX.csv')
dataY = os.path.join(sys.argv[1], 'logisticY.csv')
out = os.path.join(sys.argv[2], 'Q3a.txt')
outfile = open(out, "w")

# 3. Logistic Regression
print("################ 3. Logistic Regression ################\n", file=outfile)

trainX = np.loadtxt(dataX, delimiter=',')
trainY = np.loadtxt(dataY)

def normalize(X):
	mu = np.mean(X)
	sigma = np.std(X)
	return (X - mu)/sigma

X1 = normalize(trainX[:,0])
X2 = normalize(trainX[:,1])
X = np.column_stack((X1, X2))
X = np.column_stack((np.ones(X.shape[0]), X))
Y = trainY.reshape(-1,1)
m = len(trainY)

# (a) Logistic Regression

def sigmoid(z):
	return 1.0/(1 + np.exp(-z))

def hw(theta, x):
	return sigmoid(np.dot(x, theta))

def LLw(y, h):
	return -1*np.sum(y*np.log(h) + (1-y)*np.log(1-h))

def dJw(x, y, h):
	return np.dot(x.T, (h - y))

def Hessian(x, h):
	A = np.diag((h*(1-h)).reshape(-1,))
	return np.dot(np.dot(x.T, A), x)

def LogisticRegression(x, y):
	theta = np.zeros((x.shape[1], 1))
	h = hw(theta, x)
	prevCost = LLw(y, h)
	converged = False
	itr = 0

	while not converged:
		H = Hessian(x, h)
		theta = theta - np.dot(np.linalg.pinv(H), dJw(x, y, h))
		h = hw(theta, x)
		cost = LLw(y, h)
		error = abs(cost - prevCost)
		prevCost = cost
		itr += 1
		if error < 1e-10 or itr > 10:
			converged = True
		print('iteration {}: cost = {} error = {} '.format(itr, cost, error), end = '', file=outfile)
		print('w = {0},{1},{2}'.format(theta[0], theta[1], theta[2]), file=outfile)
	print("Final Cost =", cost, file=outfile)
	print("Final Parameters = {0},{1},{2}\n".format(theta[0], theta[1], theta[2]), file=outfile)
	return theta

theta = LogisticRegression(X, Y)
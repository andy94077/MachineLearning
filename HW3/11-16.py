import numpy as np
from DecisionTree import DecisionTree

def read_data(filename):
	'''@return: (array, array). X, Y'''
	data = np.loadtxt(filename,dtype=float)
	return data[:,:-1], data[:, -1]

trainX, trainY = read_data('hw3_train.dat')

clf = DecisionTree()
clf.fit(trainX, trainY)
print(clf)
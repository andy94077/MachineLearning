import numpy as np
import matplotlib.pyplot as plt
def read_data(path):
	data = np.loadtxt(path)
	return data[:,:-1], data[:, -1]

def draw(title=None, xlabel=None, ylabel=None, has_legend=False,savefig=False):
	if title is not None:
		plt.title(title)
	if xlabel is not None:
		plt.xlabel(xlabel)
	if ylabel is not None:
		plt.ylabel(ylabel)
	if has_legend:
		plt.legend()
	if savefig is False:
		plt.show()
	else:
		plt.savefig(savefig)
	plt.close()

def gknbor(X, trainX, trainY, k):
	return np.array([np.sign(np.sum(trainY[np.argpartition(np.sum((Xi - trainX)** 2, axis=1), k)[:k]])) for Xi in X])

def guni(X, trainX, trainY, gamma):
	return np.array([np.sign(trainY @ np.exp(-gamma*np.sum((Xi-trainX)**2,axis=1))) for Xi in X])

def err_rate(predicted, Y):
	return np.mean(predicted != Y)

if __name__ == "__main__":
	trainX, trainY = read_data('hw4_train.dat')
	testX, testY = read_data('hw4_test.dat')

	#problem 11
	ein_gknbor = [err_rate(gknbor(trainX, trainX, trainY, k), trainY) for k in range(1, 11, 2)]
	plt.plot(range(1, 11, 2), ein_gknbor, '-o')
	draw(title='Problem 11', xlabel='k', ylabel='$E_{in}(g_{k-nbor})$')
	
	#problem12
	eout_gknbor = [err_rate(gknbor(testX, trainX, trainY, k), testY) for k in range(1, 11, 2)]
	plt.plot(range(1, 11, 2), eout_gknbor, '-o')
	draw(title='Problem 12', xlabel='k', ylabel='$E_{out}(g_{k-nbor})$')

	#problem 13
	ein_guni = [err_rate(guni(trainX, trainX, trainY, gamma), trainY) for gamma in [1e-3,0.1,1,10,100]]
	plt.plot([1e-3,0.1,1,10,100], ein_guni, '-o')
	draw(title='Problem 13', xlabel='k', ylabel='$E_{in}(g_{uniform})$')

	#problem 14
	eout_guni = [err_rate(guni(testX, trainX, trainY, gamma), testY) for gamma in [1e-3,0.1,1,10,100]]
	plt.plot([1e-3,0.1,1,10,100], eout_guni, '-o')
	draw(title='Problem 14', xlabel='k', ylabel='$E_{out}(g_{uniform})$')


	
from utility import *
import numpy as np
def split_data_set(X, Y,train_n):
	'''@return: (matrix,matrix,matrix,matrix). trainX,trainY,testX,testY'''
	return X[:train_n,:], Y[:train_n,:], X[train_n:,:], Y[train_n:,:]

def bagging(trainX, trainY):
	np.random.seed()
	sample = np.random.choice(trainX.shape[0],size=trainX.shape[0])
	return trainX[sample], trainY[sample]

def err_rate_bagging(X, Y, W):
	'''@return: float
	G(x)=np.sign(np.sum(np.sign(W^T * x)))'''
	return np.float(np.sum(np.sign(np.sum(np.sign(X * W), axis=1)) != Y)) / X.shape[0]

def err_test(X, Y, W_list, ein=True, bagging=False):
	if bagging:
		E = [err_rate_bagging(X, Y, W) for W in W_list]
		print('With bagging, ',end='')
	else:
		E = [err_rate(X, Y, W) for W in W_list]
	lam = np.argmin(E)
	#print(E)
	print('minimum','Ein' if ein else 'Eout','= {}, lambda = {}'.format(E[lam], 0.05 * 10 ** lam))

if __name__ == "__main__":
	X,Y=read_data('hw2_lssvm_all.dat')
	trainX, trainY, testX, testY = split_data_set(X, Y, 400)
	lam_list = [0.05, 0.5, 5, 50, 500]
	
	w_list = [get_w(trainX, trainY, lam) for lam in lam_list]

	#problem 9
	err_test(trainX,trainY,w_list)

	#problem 10
	err_test(testX, testY, w_list,ein=False)
	
	'''
		 [ |  |  |  ...]
	W[0]=[ w1 w2 w3 ...]
		 [ |  |  |  ...]
	'''
	W_matrix_list = [np.matrix(np.concatenate([get_w(*bagging(trainX, trainY), lam=lam) for _ in range(250)], axis=1)) for lam in lam_list]

	#problem 11
	err_test(trainX, trainY, W_matrix_list, bagging=True)
	
	#problem 12
	err_test(testX, testY, W_matrix_list, ein=False, bagging=True)

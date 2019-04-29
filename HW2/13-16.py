import numpy as np
import matplotlib.pyplot as plt
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

def get_g(data, u):
	min_g = (-1, 0, -np.inf)
	min_err_sum = np.sum(u)
	data=data.copy()
	for i in range(data.shape[1] - 1):  #exclude Y
		order=data[:, i].argsort()
		data = data[order]  #sort by the ith feature
		u = u[order]
		Xi = data[:,i]
		Y = data[:, -1]
		
		thetas = np.concatenate(([-np.inf], (Xi[:-1] + Xi[1:]) / 2))
		for s in [-1, 1]:
			for theta in thetas:
				err_sum = u@(g_func((s,i,theta),data)!=Y)
				if err_sum < min_err_sum:
					min_err_sum = err_sum
					min_g = (s, i, theta)
	return min_g, min_err_sum / data.shape[0], min_err_sum/np.sum(u)

def g_func(tup,X):
	s, i, theta = tup
	return s * np.sign(X[:, i] - theta)

def G_func(G, X):
	return np.sign(np.sign(X[:, G[:, 1].astype(int)] - G[:, 2]) @ G[:, 0:1]).flatten()

def err_rate(X, Y, G):
	return np.float(np.sum(G_func(G, X) != Y)) / X.shape[0]

if __name__ == "__main__":
	train = np.loadtxt('hw2_adaboost_train.dat', dtype=np.float64)
	trainX, trainY = train[:,:-1], train[:, -1]
	test = np.loadtxt('hw2_adaboost_test.dat', dtype=np.float64)
	testX, testY = test[:,:-1], test[:, -1]

	T = 300
	Ein = []
	G = np.zeros((T,3)) #[s, i, theta]
	Ein_G=[]
	U = []
	u = np.full(train.shape[0], 1 / train.shape[0])
	Eout_G=[]
	for t in range(T):
		g, ein, epsilon = get_g(train, u)

		Ein.append(ein)
		
		G[t] = np.array([g[0] * np.log(1 / epsilon - 1) / 2, g[1], g[2]])
		Ein_G.append(err_rate(trainX, trainY, G))
		
		U.append(np.sum(u))
		u *= np.sqrt(1 / epsilon - 1)**(-trainY * g_func(g, trainX))
		Eout_G.append(err_rate(testX, testY, G))

	#problem 13
	print('Ein(gT) =',Ein[-1])
	plt.plot(range(T), Ein)
	draw(title='Problem 13',xlabel='t',ylabel='$E_{in}(g_t)$',savefig='13.jpg')

	#problem 14
	print('Ein(GT) =',Ein_G[-1])
	plt.plot(range(T), Ein_G)
	draw(title='Problem 14',xlabel='t', ylabel='$E_{in}(G_t)$',savefig='14.jpg')
	
	#problem 15
	print('UT =', U[-1])
	plt.plot(range(T), U)
	draw(title='Problem 15',xlabel='t', ylabel='$U_t$',savefig='15.jpg')
	
	#problem 16
	print('Eout(GT) =', Eout_G[-1])
	#print(np.argmin(Eout_G))
	plt.plot(range(T), Eout_G)
	draw(title='Problem 16',xlabel='t', ylabel='$E_{out}(G_t)$',savefig='16.jpg')
	
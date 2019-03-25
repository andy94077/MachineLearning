from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
	data=np.loadtxt(filename)
	X = data[:, 1:]
	y = data[:, 0]
	return X, y

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

def problem13(X,y):
	print('problem 13')
	w_list = []
	y_bin = (y == 2.0) * 2 - 1  #turn True,False into 1,-1
	for c in [1e-5,1e-3,1e-1,1e1,1e3]:
		print('c=%f'%c)
		clf = SVC(C=c,kernel='linear')
		clf.fit(X,y_bin) 
		#print(clf.coef_)
		w_list.append(np.linalg.norm(clf.coef_))

	print(w_list)
	plt.plot([-5, -3, -1, 1, 3], w_list)
	draw(xlabel='log C',ylabel='||w||')


def problem14(X,y):
	print('problem 14')
	ein_list = []
	Z = np.concatenate((np.ones((X.shape[0], 1)), np.sqrt(2) * X, X ** 2), axis=1)
	y_bin=(y == 4.0) * 2 - 1 #turn True,False into 1,-1
	for c in [1e-5,1e-3,1e-1,1e1,1e3]:
		print('c=%f'%c)
		clf = SVC(kernel='poly', C=c, degree=2, coef0=1, gamma=1)
		clf.fit(X, y_bin)
		
		ein_list.append(1.0- clf.score(X, y_bin))

	print(ein_list)
	plt.plot([-5, -3, -1, 1, 3], ein_list)
	draw(xlabel='log C',ylabel='Ein')

def problem15(X,y):
	print('problem 15')
	def K(x, x_):
		return np.exp(-80 * np.sum((x - x_)** 2,axis=1,keepdims=True))

	distance_list = []
	y_bin = (y == 0.0) * 2 - 1
	for c in [1e-2,1e-1,1e0,1e1,1e2]:
		print('c=%f' % c)
		clf = SVC(C=c, gamma=80)
		clf.fit(X, y_bin)
		
		alpha_y = clf.dual_coef_
		w_len = np.sqrt(np.sum(alpha_y.T @ alpha_y * np.concatenate([K(X[i], clf.support_vectors_).T for i in clf.support_], axis=0)))
		distance_list.append(1/w_len)

	print(distance_list)
	plt.plot(list(range(-2, 3)), distance_list)
	draw(xlabel='log C', ylabel='distance')


#16
def get_hist(tup):
	hist = np.zeros(5)
	data = tup[0]
	process_n = tup[1]
	np.random.seed()
	for i in range(100//process_n):
		print(i,end=' ',flush=True)
		np.random.shuffle(data)
		test_X, test_y = data[:1000,:-1], data[:1000, -1]
		train_X, train_y = data[1000:,:-1], data[1000:, -1]

		min_eval,min_eval_log_gamma=1.0,1e-2
		for log_gamma in range(-2,3):
			clf = SVC(C=0.1, gamma=10**log_gamma)
			clf.fit(train_X, train_y)
			
			Eval=1.0 - clf.score(test_X, test_y)
			if Eval < min_eval:
				min_eval = Eval
				min_eval_log_gamma = log_gamma
			
		hist[min_eval_log_gamma + 2] += 1  #shift index to 0
	return hist

def problem16(X,y):
	print('problem 16')
	from multiprocessing import Pool
	y_bin = (y == 0.0) * 2 - 1
	data = np.concatenate((X, y_bin.reshape(-1, 1)), axis = 1)
	process_n=10
	assert 100%process_n==0
	poo=Pool(process_n)

	argument_list=[(data.copy(),process_n) for _ in range(process_n)]
	total = sum(poo.map(get_hist, argument_list))
	plt.bar(list(range(-2,3)),total)
	draw(xlabel='log gamma')

if __name__ == "__main__":
	X, y = read_data('features.train')
	problem13(X,y)
	problem14(X,y)
	problem15(X, y)
	problem16(X, y)

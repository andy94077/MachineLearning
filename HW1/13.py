from sklearn.svm import SVC,LinearSVC
import numpy as np
import matplotlib.pyplot as plt

def read_data(filename):
	data=np.loadtxt(filename)
	X = data[:, 1:]
	y = data[:, 0]
	return X, y

def draw(title=None, xlabel=None, ylabel=None):
	if title is not None:
		plt.title(title)
	if xlabel is not None:
		plt.xlabel(xlabel)
	if ylabel is not None:
		plt.ylabel(ylabel)
	plt.legend()
	plt.show()


X, y=read_data('features.train')

#13
'''
w_list=[]
for c in [1e-5,1e-3,1e-1,1e1,1e3]:
	print('c=%f'%c)
	clf = SVC(C=c,kernel='linear') #LinearSVC(C=c,loss='hinge')
	clf.fit(X,(y==2.0)*2-1) #turn True,False into 1,-1
	print(clf.coef_)
	w_list.append(np.linalg.norm(clf.coef_))#np.sum(alpha*y[clf.support_]*X[clf.support_,:].T,axis=0))

print(w_list)
plt.plot([-5, -3, -1, 1, 3], w_list)
draw(xlabel='log C',ylabel='||w||')
'''


#14
ein_list=[]
ones = np.ones((X.shape[0], 1))
'''
Z = [[--z1--]
	 [--z2--]
	 [......]]
'''
Z = np.concatenate((ones, np.sqrt(2) * X, X * X), axis=1)
y_bin=(y == 4.0) * 2 - 1 #turn True,False into 1,-1
for c in [1e-5,1e-3,1e-1,1e1,1e3]:
	print('c=%f'%c)
	clf = LinearSVC(C=c,loss='hinge')#SVC(kernel='poly', C=c, degree=2, coef0=1, gamma=1)
	clf.fit(Z, y_bin)
	
	w=clf.coef_.T#w = np.sum(clf.dual_coef_.reshape(-1,1) * Z[clf.support_], axis=0)
	ein_list.append(np.sum(y_bin!=np.sign(Z@w+clf.intercept_).T)/X.shape[0])

print(ein_list)
plt.plot([-5, -3, -1, 1, 3], ein_list)
draw(xlabel='log C',ylabel='Ein')


'''
#15
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
	distance_list.append(np.abs(np.sum(alpha_y.T * K(clf.support_vectors_[0],clf.support_vectors_)+float(clf.intercept_)))/w_len)

plt.plot(list(range(-2, 3)), distance_list)
draw(xlabel='log C', ylabel='distance')
'''

'''
#16
from multiprocessing import pool
data = np.concatenate((X, y.reshape(-1, 1)), axis = 1)

def get_hist()
hist = {int(10 ** i): 0 for i in range(4)}
for i in range(100):
	print(i,end=' ',flush=True)
	np.random.shuffle(data)
	test_X, test_y = data[:1000,:-1], data[:1000, -1]
	train_X, train_y = data[1000:,:-1], data[1000:, -1]

	min_eval,min_eval_gamma=1.0,1e-2
	for gamma in [1e-2,1e-1,1e0,1e1,1e2]:
		clf = SVC(C=0.1, gamma=gamma)
		clf.fit(train_X, train_y)
		
		Eval=clf.score(test_X, test_y)
		if Eval < min_eval:
			min_eval = Eval
			min_eval_gamma = gamma
		
	hist[int(100 * min_eval_gamma)] += 1

plt.hist(hist.values(),bins=[-2,-1,0,1,2])
draw(xlabel='log gamma')
'''
'''
print(clf.support_)
print(np.abs(clf.dual_coef_))
print(clf.support_vectors_)
print(clf.intercept_)
'''
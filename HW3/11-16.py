import pickle
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from DecisionTree import DecisionTree

def read_data(filename):
	'''@return: (array, array). X, Y'''
	data = np.loadtxt(filename,dtype=float)
	return data[:,:-1], data[:, -1].astype(int)

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


def problem13(clf,trainX, trainY, testX, testY):
	Ein, Eout = [], []
	pruned = DecisionTree()

	for h in range(1, clf.height):
		pruned = pruned.fit(trainX,trainY,h)
		Ein.append(pruned.score(trainX, trainY))
		Eout.append(pruned.score(testX, testY))
	
	plt.plot(range(1,clf.height+1),Ein+[clf.score(trainX,trainY)])
	draw(title='$E_{in}(g_h)\;v.s.\;h$',xlabel='h',ylabel='$E_{in}$')
	plt.plot(range(1,clf.height+1),Eout+[clf.score(testX,testY)])
	draw(title='$E_{out}(g_h)\;v.s.\;h$',xlabel='h',ylabel='$E_{out}$')

def bagging(X, Y, rate):
	np.random.seed()
	choice = np.random.choice(np.arange(X.shape[0]), size=int(rate * X.shape[0]))
	return X[choice], Y[choice]

def generate_tree(tup):
	'''@parameters:
		tup: (trainX, trainY, testX)
	@return: (DecisionTree, array, array, array): tree, prediction, Ein, out_prediction'''
	trainX, trainY = tup[0], tup[1]
	bagX, bagY = bagging(trainX, trainY, 0.8)
	testX = tup[2]
	
	tree = DecisionTree().fit(bagX, bagY)
	prediction = tree.predict(trainX)
	return tree, prediction, np.mean(prediction != trainY), tree.predict(testX)

def random_forest(tree_n, trainX, trainY, testX,has_model=False):
	if has_model:
		with open('model','rb') as f:
			return pickle.load(f)
	else:
		poo = Pool(32)
		return_list = poo.map(generate_tree, [(trainX, trainY, testX)] * tree_n)

		trees = [tup[0] for tup in return_list]
		predictions = np.array([tup[1] for tup in return_list])
		Ein = [tup[2] for tup in return_list]
		out_predictions = np.array([tup[3] for tup in return_list])

		with open('model','wb') as f:
			pickle.dump((trees, predictions, Ein, out_predictions), f)
		return trees, predictions, Ein, out_predictions


if __name__ == "__main__":
	trainX, trainY = read_data('hw3_train.dat')#read_data('test')
	testX, testY = read_data('hw3_test.dat')

	clf = DecisionTree()
	clf.fit(trainX, trainY)

	#problem 12
	print('Ein =', clf.score(trainX, trainY))
	print('Eout =', clf.score(testX, testY))

	#problem13(clf, trainX, trainY, testX, testY)
	
	T=30000
	trees, predictions, Ein, out_predictions = random_forest(T, trainX, trainY, testX)

	#problem 14
	plt.hist(Ein)
	draw(title='$E_{in}(g_t)$',savefig='14.jpg')

	#problem 15
	EinG = np.mean(np.sign(np.cumsum(predictions, axis=0)) != trainY, axis=1)
	plt.plot(range(1,T+1), EinG)
	draw(title='$E_{in}(G_t)$',savefig='15.jpg')
	
	#problem 16
	EoutG = np.mean(np.sign(np.cumsum(out_predictions, axis=0)) != testY, axis=1)
	plt.plot(range(1,T+1), EoutG)
	draw(title='$E_{out}(G_t)$',savefig='16.jpg')
	

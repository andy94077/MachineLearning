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

def k_means(X, k):
	mus = X[np.random.choice(np.arange(X.shape[0]), k, replace=False)]
	group = [[] for _ in range(k)]  
	for i, Xi in enumerate(X):
		group[np.argmin(np.sum((Xi - mus)** 2, axis=1))].append(i)
	
	while True:
		mus = np.array([np.mean(X[g], axis=0) for g in group])
		group2 = [[] for _ in range(k)]
		for i, Xi in enumerate(X):
			group2[np.argmin(np.sum((Xi - mus)** 2, axis=1))].append(i)
		if group == group2:
			break
		else:
			group = group2
	
	# Ein
	return np.sum([np.sum((X[g] - mus[i])** 2) for i, g in enumerate(group)]) / X.shape[0]
	
if __name__ == "__main__":
	X = np.loadtxt('hw4_nolabel_train.dat')
	
	eins = np.array([[k_means(X, k) for _ in range(500)] for k in [2, 4, 6, 8, 10]])
	ave_eins = np.mean(eins,axis=1)
	var_eins = np.var(eins, axis=1)
	
	plt.plot(range(2, 12, 2), ave_eins, '-o')
	draw(title='Problem 15', xlabel='k', ylabel='$\mathbb{E}[E_{in}]$')
	
	plt.plot(range(2, 12, 2), var_eins, '-o')
	draw(title='Problem 16', xlabel='k', ylabel='$VAR[E_{in}]$')
	
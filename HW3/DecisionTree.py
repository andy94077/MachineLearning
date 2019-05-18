from queue import Queue
import numpy as np

class DecisionTree():
	def __init__(self):
		self.fitted = False
		self.b = (0,0,0) #(s,i,theta)
		self.left = -1
		self.right = 1
	
	def __str__(self):
		string = ''
		preorder = [self]
		while preorder:
			item = preorder.pop()
			if isinstance(item, DecisionTree):
				string += '(%d, %d, %.3f)\n' % item.b
				preorder.append(item.right)
				preorder.append(item.left)
			else:
				string += str(item) + '\n'
		return string
	
	def g_func(self, b, X):
		return (b[0] * np.sign(X[:, b[1]] - b[2])).ravel()

	def err_gini(self, b, X, Y):
		result = self.g_func(b, X)
		D = [Y[result == -1], Y[result == 1]]
		return sum([(Dc.shape[0] - (np.sum(Dc==-1)**2+np.sum(Dc==1)**2)/Dc.shape[0]) for Dc in D if Dc.shape[0]!=0])

	def find_b(self, X, Y):
		if np.all(Y == Y[0]):
			return (int(Y[0]), 0, -np.inf),True  #i does not matter
		elif np.all(X == X[0]):  #all X are the same
			return (int(Y[0]), 0, -np.inf),True  #return garbage
		else:
			min_err = np.inf
			min_err_b = (Y[0], 0, -np.inf)
			for i in range(X.shape[1]):
				Xi = np.sort(X[:, i])
				thetas = np.concatenate(([-np.inf], (Xi[:-1] + Xi[1:]) / 2))
				for s in [-1, 1]:
					for theta in thetas:
						err = self.err_gini((s, i, theta), X, Y)
						if err < min_err:
							min_err = err
							min_err_b = (s, i, theta)
			return min_err_b, min_err_b[2]==-np.inf # if min_err_b[2]==-np.inf, it means 'b' cannot split X into 2 partition
							

	def fit(self, X, Y):
		self.b, end = self.find_b(X, Y)
		#print(self.b)
		if not end:
			result = self.g_func(self.b, X)
			self.left = DecisionTree().fit(X[result == -1], Y[result == -1])
			self.right = DecisionTree().fit(X[result == 1], Y[result == 1])
		self.fitted = True
		return self

	def predict(self, X):
		if not self.fitted:
			raise 'Not fitted yet'
		else:
			result = []
			for x in X:
				node = self
				while isinstance(node,DecisionTree):
					node = node.left if node.g_func(self.b, x.reshape(1, -1)) == -1 else node.right
				result.append(node)
			return np.array(result)
	
	def score(self, X, Y):
		return np.mean(self.predict(X) != Y)
		
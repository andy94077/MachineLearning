import numpy as np
import matplotlib.pyplot as plt

train = np.loadtxt('hw2_adaboost_train.dat', dtype=np.float64)
trainX, trainY = train[:,:-1], train[:, -1]

plt.scatter(trainX[:, 0], trainX[:, 1], c=['b' if y == 1 else 'r' for y in trainY])
plt.show()
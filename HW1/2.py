from sklearn.svm import SVC
import numpy as np

X = np.array([[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]])
y = np.array([-1, -1, -1, 1, 1, 1, 1])

svclassifier = SVC(kernel='poly',C=1e10, degree=2, coef0=1, gamma=1)
svclassifier.fit(X, y)

print(svclassifier.support_)
print(np.abs(svclassifier.dual_coef_).shape)
print(svclassifier.support_vectors_)
print(svclassifier.intercept_)
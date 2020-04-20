from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
from mpl_toolkits.mplot3d import Axes3D, art3d
import mglearn
import matplotlib.pyplot as plt
import numpy as np

X, y = make_blobs(centers=4, random_state=8)
X_new = np.hstack([X, X[:, 1:] ** 2])
y = y % 2

linear_svm = LinearSVC()
linear_svm.fit(X_new, y)
coef, intercept = linear_svm.coef_.ravel(), linear_svm.intercept_

# mglearn.plots.plot_2d_separator(linear_svm, X)
# mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()

figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:, 0].min() -2 , X_new)
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60)
ax.set_xlabel("feature0")
ax.set_ylabel("feature1")
ax.set_zlabel("feature1 ** 2")
plt.show()

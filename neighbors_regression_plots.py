import mglearn
import matplotlib.pyplot as plt

mglearn.plots.plot_knn_regression(n_neighbors=1)
mglearn.plots.plot_knn_regression(n_neighbors=3)
mglearn.plots.plot_knn_regression(n_neighbors=9)

plt.show()

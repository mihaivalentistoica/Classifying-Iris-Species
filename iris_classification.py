from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mglearn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np

iris_data_set = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_data_set['data'], iris_data_set['target'], random_state=0)
print(iris_data_set.keys())
# iris_data_frame = pd.DataFrame(X_train, columns=iris_data_set["feature_names"])
# grr = pd.plotting.scatter_matrix(iris_data_frame, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60,
#                                  alpha=.8, cmap=mglearn.cm3)
# plt.show()

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
# print(knn)
X_new = np.array([[5, 2.9, 1, 0.2]])

prediction = knn.predict(X_new)
print('Prediction: ', prediction)
print(iris_data_set['target_names'][prediction])

y_pred = knn.predict(X_test)
print(y_pred)
accuracy = np.mean(y_pred == y_test)

print(f'Test set score: {accuracy:.4f}')

print(f'Test result from class: {knn.score(X_test, y_test):.4fF}')

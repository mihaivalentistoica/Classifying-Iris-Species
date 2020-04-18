from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import mglearn
import mglearn.datasets as datasets

# print(datasets.maek)
X, y = datasets.make_forge()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)

test_predictions = clf.predict(X_test)

print(f'Test set prediction: {test_predictions}')
print(f'Test score: {clf.score(X_test, y_test):.4f}')

fig, axis = plt.subplots(1, 3, figsize=(10, 3))
for ng_neighbors, ax in zip([1, 3, 9], axis):
  clf2 = KNeighborsClassifier(n_neighbors=ng_neighbors).fit(X, y)
  mglearn.plots.plot_2d_separator(clf2, X, fill=True, eps=0.5, ax= ax, alpha=.4)
  mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)

axis[0].legend(loc=3)
plt.show()

# for n_neighbors, ax in zip([1, 3, 9], axes):
# fix, axes = plt.subplots(1, 3, figsize=(10, 3))

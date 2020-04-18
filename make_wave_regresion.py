from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import mglearn.datasets as data_sets

X, y = data_sets.make_wave(n_samples=40)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, y_train)
print(f'Test set prediction: \n{reg.predict(X_test)}')
print(f'Test regression score: {reg.score(X_test, y_test)}')

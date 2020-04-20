from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from decision_tree import plot_feature_importances

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
forest = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)
forest.fit(X_train, y_train)

print(f'Score on training set: {forest.score(X_train, y_train):.3f}')
print(f'Score on test data set: {forest.score(X_test, y_test):.3f}')

plot_feature_importances(forest)

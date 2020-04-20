from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

gradient = GradientBoostingClassifier(random_state=0, learning_rate=0.01)
gradient.fit(X_train, y_train)

print(f'Training set score: {gradient.score(X_train, y_train):.3f}')
print(f'Testing set score: {gradient.score(X_test, y_test):.3f}')
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
log_reg = LogisticRegression(max_iter=100000, C=1)
log_reg.fit(X_train, y_train)

print(f'Training set score: {log_reg.score(X_train, y_train):.3f}')
print(f'Test set score: {log_reg.score(X_test, y_test):.3f}')

log_reg100 = LogisticRegression(max_iter=100000, C=100)
log_reg100.fit(X_train, y_train)

print(f'Training set score: {log_reg100.score(X_train, y_train):.3f}')
print(f'Test set score: {log_reg100.score(X_test, y_test):.3f}')

log_reg001 = LogisticRegression(max_iter=100000, C=.001)
log_reg001.fit(X_train, y_train)

print(f'Training set score: {log_reg001.score(X_train, y_train):.3f}')
print(f'Test set score: {log_reg001.score(X_test, y_test):.3f}')

plt.plot(log_reg.coef_.T, 'o', label="C=1")
plt.plot(log_reg100.coef_.T, 'o', label="C=100")
plt.plot(log_reg001.coef_.T, 'o', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.legend()
plt.show()
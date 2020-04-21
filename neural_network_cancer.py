from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

cancer = load_breast_cancer()

# print(f'Maximum cancer data per feature: \n {cancer.data.max(axis=0)}')
# print(cancer.keys())

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)

# Pentru a creste acuratetea retelei trebuie sa incadram valorile inre 0 si 1

# Calculam media caracteristicilor
mean_on_train = X_train.mean(axis=0)
# Calcula deviatia standar a fiecarei caracteristici
std_on_train = X_train.std(axis=0)

X_train_scaled = (X_train - mean_on_train) / std_on_train
print(X_train_scaled[0])
X_test_scaled = (X_test - mean_on_train) / std_on_train

mlp = MLPClassifier(random_state=0, max_iter=1000, alpha=1)
mlp.fit(X_train_scaled, y_train)

print(f'Accuracy on train data: {mlp.score(X_train_scaled, y_train):.4f}')
print(f'Accuracy on test data set {mlp.score(X_test_scaled, y_test):4f}')

plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
plt.yticks(range(30), cancer.feature_names)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()
plt.show()
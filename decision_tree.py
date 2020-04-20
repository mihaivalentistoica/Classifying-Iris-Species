from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importances(model):
  n_features = cancer.data.shape[1]
  plt.barh(range(n_features), model.feature_importances_, align='center')
  plt.yticks(np.arange(n_features), cancer.feature_names)
  plt.xlabel('Feature importance')
  plt.ylabel('Feature')
  plt.show()


cancer = load_breast_cancer()
print(f'Feature names: \n{cancer.feature_names}')
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(max_depth=4, random_state=0)
tree.fit(X_train, y_train)
print(f'Accuracy on training set: {tree.score(X_train, y_train):.3f}')
print(f'Accuracy on test set: {tree.score(X_test, y_test):.3f}')
export_graphviz(tree, out_file="tree.dot", class_names=['malignant', 'benign'], feature_names=cancer.feature_names,
                impurity=False, filled=True)

print(f'Feature importances: \n{tree.feature_importances_}')
plot_feature_importances(tree)

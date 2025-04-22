import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Load the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Implement Random Forest manually
n_trees = 100
n_samples = X_train.shape[0]
trees = []


# Train n_trees Decision Trees
for _ in range(n_trees):
    indices = np.random.choice(n_samples, n_samples, replace=True)
    X_bootstrap = X_train.iloc[indices]  # No change needed for X_bootstrap
    y_bootstrap = y_train[indices]  # Directly index y_train with indices

    tree = DecisionTreeClassifier(random_state=42)
    tree.fit(X_bootstrap, y_bootstrap)
    trees.append(tree)


# Function to predict using the Random Forest
def random_forest_predict(X):
    tree_predictions = np.array([tree.predict(X) for tree in trees])
    final_predictions = []

    for i in range(X.shape[0]):
        mode_result = np.bincount(tree_predictions[:, i]).argmax()
        final_predictions.append(mode_result)

    return np.array(final_predictions)

# Predict using the manually implemented Random Forest
y_pred_rf = random_forest_predict(X_test)

# Calculate accuracy
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print("Predictions: ", y_pred_rf)
print("Count of class 0 = ", np.sum(y_pred_rf == 0))
print("Count of class 1 = ", np.sum(y_pred_rf == 1))
print("Count of class 2 = ", np.sum(y_pred_rf == 2))
print(f"Manually Implemented Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

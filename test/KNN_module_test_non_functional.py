
#%% my KNN module
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import sklearn.metrics  
from sklearn.metrics import mean_absolute_error

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))


class KNN:

    def __init__(self,k=3):
        self.k = k


    def fit(self,X,y):
        self.X_train = X
        self.y_train = y


    def nearest_neighbor_indices(self ,x):
        
        # compute distnaces
        distances = [euclidean_distance(x,x_train) for x_train in self.X_train]

        # get k nearest smaples, labels
        k_indices = np.argsort(distances)[:self.k]

        return k_indices


    def _predict(self , x):
       
        # get k nearest smaples, labels
        k_indices = self.nearest_neighbor_indices(x)
        k_nearest_labels = [self.y_train[i] for i in k_indices]


        # majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
    






#%% Running Test using iris dataset

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate and fit scikit-learn's KNN classifier
sklearn_knn = KNeighborsClassifier(n_neighbors=3)
sklearn_knn.fit(X_train, y_train)

# Instantiate and fit my KNN classifier
my_knn = KNN(k=3)
my_knn.fit(X_train, y_train)

# Predict labels using scikit-learn's KNN classifier
sklearn_predictions = sklearn_knn.predict(X_test)

# Predict labels using your KNN classifier
my_predictions = []
for x_test in X_test:
    my_predictions.append(my_knn._predict(x_test))


# Compare predictions
print("Scikit-learn predictions:")
print(sklearn_predictions)

print("\nMy KNN predictions:")
print(my_predictions)

# Check if predictions match
print("\nPredictions match:", np.array_equal(sklearn_predictions, my_predictions))




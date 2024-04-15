import numpy as np
import sys

sys.path.append('C:\\Users\\seyed\\Desktop\\RecommSys\\src') # Replace the path
from modules import KNN 

# Initialize test data
X_train = np.array([[1, 2], [3, 4], [5, 6]])
y_train = np.array([0, 1, 0])
knn = KNN(k=2)
knn.fit(X_train, y_train)

def test_nearest_neighbor_indices():
    # Test nearest_neighbor_indices method
    test_point = np.array([0, 0])
    expected_indices = [0, 1]
    assert knn.nearest_neighbor_indices(test_point) == expected_indices


def test_predict():
    # Test _predict method
    test_point = np.array([0, 0])
    expected_prediction = 0
    assert knn._predict(test_point) == expected_prediction
import numpy as np
from typing import Self
import math

"""
This is a suggested template and you do not need to follow it. You can change any part of it to fit your needs.
There are some helper functions that might be useful to implement first.
At the end there is some test code that you can use to test your implementation on synthetic data by running this file.
"""

def count(y: np.ndarray) -> np.ndarray:
    """
    Count unique values in y and return the proportions of each class sorted by label in ascending order.
    Example:
        count(np.array([3, 0, 0, 1, 1, 1, 2, 2, 2, 2])) -> np.array([0.2, 0.3, 0.4, 0.1])
    """
    #np.unique() returns the unique value sorted and a frequency array
    N = len(y)
    unique, counts = np.unique(y, return_counts=True)
    proportions_sorted = []
    for i in range(len(counts)):
        prop = counts[i] / N
        proportions_sorted.append(prop)
    return np.array(proportions_sorted)


def gini_index(y: np.ndarray) -> float:
    """
    Return the Gini Index of a given NumPy array y.
    The forumla for the Gini Index is 1 - sum(probs^2), where probs are the proportions of each class in y.
    Example:
        gini_index(np.array([1, 1, 2, 2, 3, 3, 4, 4])) -> 0.75
    """
    y_probs = count(y)
    return 1 - np.sum(y_probs ** 2)



def entropy(y: np.ndarray) -> float:
    """
    Return the entropy of a given NumPy array y.
    """
    prob_sorted_by_label = count(y)
    result = 0
    for p in prob_sorted_by_label:
        if p>0:
            result += p * math.log2(p)
    
    return -result


def split(x: np.ndarray, value: float) -> np.ndarray:
    """
    Return a boolean mask for the elements of x satisfying x <= value.
    Example:
        split(np.array([1, 2, 3, 4, 5, 2]), 3) -> np.array([True, True, True, False, False, True])
    """
    return x <= value


def most_common(y: np.ndarray) -> int:
    """
    Return the most common element in y.
    Example:
        most_common(np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])) -> 4
    """
    #returns one array , unique, with unique values and another , counts, with frequency of each value
    unique, counts = np.unique(y, return_counts=True)
    #returns the index of the largest value. The most common value is max of the frequency array.
    indx_max = np.argmax(counts)
    return unique[indx_max]

def numb_of_features(value: str, features: int) -> int:
    """
    Returns the number of features after passing the chosen max_feature function.
    """
    value = value.lower()
    if value not in ["sqrt", "log2"]:
        raise ValueError("Function not recognized. Please use 'sqrt' or 'log2'.")
    elif features == 0:
        return 0
    else:
        func = math.log2 if value == "log2" else math.sqrt
        return max(1, math.floor(func(features)))
    

def find_best_splits(X: np.ndarray, y: np.ndarray, rng: np.random.Generator, criterion: str = "entropy", max_features: str | None = None) -> tuple:
    """
    Returns a tuple containing the feature with the most information gain along with the subsets of X and y
    """
    if criterion.lower() not in ["entropy", "gini"]:
        raise ValueError(f"{criterion} is not a valid function. Please use 'entropy' or 'gini'.")
    
    num_features = 0
    if max_features is not None:
        num_features = numb_of_features(max_features, features=X.shape[1])

    features_list = range(X.shape[1]) if max_features is None else rng.choice(X.shape[1], num_features, replace=False)

    impurity_func = entropy if criterion == "entropy" else gini_index
    best_feature_idx = None
    best_IG = -math.inf
    best_X_left, best_X_right, best_y_left, best_y_right = None, None, None, None

    for feature in features_list:
        feature_mean = np.mean(X[:, feature])
        feature_boolean_mask = split(X[:, feature], feature_mean)
        X_left, y_left = X[feature_boolean_mask], y[feature_boolean_mask] # Subset of X,y  for x <= feature_mean
        X_right, y_right = X[~feature_boolean_mask], y[~feature_boolean_mask] # Subset of X,y for x > feature_mean

        # entropy for both subset of y
        entropy_left = impurity_func(y_left)
        entropy_right = impurity_func(y_right)
        
        # weights of right and left
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)

        conditional_entropy = (weight_left * entropy_left) + (weight_right * entropy_right)
        
        IG = impurity_func(y) - conditional_entropy
        # Keeping tab of the values that maximizing information gain
        if(IG > best_IG):
            best_feature_idx = feature
            best_IG = IG
            best_X_left, best_y_left = X_left, y_left
            best_X_right, best_y_right = X_right, y_right

    return best_feature_idx, best_X_left, best_y_left, best_X_right, best_y_right


class Node:
    """
    A class to represent a node in a decision tree.
    If value != None, then it is a leaf node and predicts that value, otherwise it is an internal node (or root).
    The attribute feature is the index of the feature to split on, threshold is the value to split at,
    and left and right are the left and right child nodes.
    """

    def __init__(
        self,
        feature: int = 0,
        threshold: float = 0.0,
        left: int | Self | None = None,
        right: int | Self | None = None,
        value: int | None = None,
    ) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        # Return True iff the node is a leaf node
        return self.value is not None
    

class DecisionTree:
    def __init__(
        self,
        max_depth: int | None = None,
        criterion: str = "entropy",
        max_features: str | None = None,
        random_state: int = 0
    ) -> None:
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.rng = np.random.default_rng(random_state)

    def fit(self,
        X: np.ndarray,
        y: np.ndarray,
    ):
        self.root = self._fit(X,y, 0)

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int
    ):
        """
        This functions learns a decision tree given (continuous) features X and (integer) labels y.
        """
        #Checking if datapoints have the same label
        same_label = len(np.unique(y)) == 1
       
        if(same_label):
            return Node(value=y[0])
        #Checking if max depth is reached or if identical feature values
        if(self.max_depth is not None):
            if(depth >= self.max_depth):
                return Node(value=most_common(y))
        if ( all(np.all(X[i] == X[0]) for i in range(len(X)))):
            return Node(value=most_common(y))
        
        #Finding the best split
        best_feature_idx, best_X_left, best_y_left, best_X_right, best_y_right = find_best_splits(X,y, rng=self.rng, criterion=self.criterion, max_features=self.max_features)

        #No valid split is found
        if len(best_X_left) == 0 or len(best_X_right) == 0:
            return Node(value=most_common(y))

        node = Node(feature=best_feature_idx, threshold= np.mean(X[:, best_feature_idx]))

        node.left = self._fit(best_X_left, best_y_left, depth+1)

        node.right = self._fit(best_X_right, best_y_right, depth+1)
        return node
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given a NumPy array X of features, return a NumPy array of predicted integer labels.
        """
        predictions = []
        for i in range(len(X)):
            predictions.append(self._predict(self.root, X[i]))
        return np.array(predictions)
    
    def _predict(self, node: Node, features):
        result = node.value
        indx = node.feature
        if(not node.is_leaf()):
            if(features[indx] <= node.threshold):
                result = self._predict(node.left, features)
            else:
                result = self._predict(node.right, features)
        return result
        
        
    
    def print_tree(self):
        if self.root is not None:
            self._print_tree(self.root)
        else:
            print("The tree hasn't been trained yet.")

    def _print_tree(self, node: Node, level=0):
        """
        Prints the given tree.
        """

        if(node.is_leaf()):
            print(" " * level + f"Value: {node.value}")
        else:
            print(" " * level + f"feature: {node.feature}, threshold: {node.threshold}")
        
            self._print_tree(node.left, level+1)

            self._print_tree(node.right, level+1)

if __name__ == "__main__":
    # Test the DecisionTree class on a synthetic dataset
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    seed = 0

    np.random.seed(seed)

    X, y = make_classification(
        n_samples=100, n_features=10, random_state=seed, n_classes=2
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=seed, shuffle=True
    )

    # Expect the training accuracy to be 1.0 when max_depth=None
    
    rf = DecisionTree(max_depth=None, criterion="entropy")
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")
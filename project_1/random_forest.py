import numpy as np
from decision_tree import DecisionTree, most_common

class RandomForest:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        criterion: str = "entropy",
        max_features: None | str = "sqrt",
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.criterion = criterion
        self.max_features = max_features
        self.decision_trees = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        for tree in range(self.n_estimators):
            tree = DecisionTree(max_depth=self.max_depth, criterion=self.criterion, max_features=self.max_features)
            """
            For each Tree: Add X and Y. Draw a random subset of the data with replacement. Split X and Y.
            Train the tree on this data. 
            """
            num_rows = X.shape[0]
            random_indices = np.random.choice(num_rows, num_rows, replace=True)
            X_subset, y_subset = X[random_indices], y[random_indices]
            tree.fit(X_subset, y_subset)
            self.decision_trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Implement this function"
        )  # Remove this line when you implement the function


if __name__ == "__main__":
    # Test the RandomForest class on a synthetic dataset
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

    rf = RandomForest(
        n_estimators=20, max_depth=5, criterion="entropy", max_features="sqrt"
    )
    rf.fit(X_train, y_train)

    print(f"Training accuracy: {accuracy_score(y_train, rf.predict(X_train))}")
    print(f"Validation accuracy: {accuracy_score(y_val, rf.predict(X_val))}")

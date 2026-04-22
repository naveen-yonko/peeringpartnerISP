import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    """
    Random Forest built from scratch.
    Trains multiple Decision Trees on random subsets of data and features,
    then takes majority vote for prediction.
    """

    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2,
                 min_samples_leaf=1, max_features="sqrt"):
        """
        n_trees          : number of decision trees to build
        max_depth        : max depth of each tree
        min_samples_split: min samples to split a node
        min_samples_leaf : min samples at a leaf
        max_features     : number of features to consider at each split
                           "sqrt" = sqrt(n_features), "log2" = log2(n_features)
        """
        self.n_trees           = n_trees
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.max_features      = max_features
        self.trees             = []   # stores (tree, feature_indices) tuples

    # ── Bootstrap Sample ─────────────────────────────────────────────────────
    def _bootstrap_sample(self, X, y):
        """
        Random sampling WITH replacement
        Each tree trains on a different bootstrap sample
        """
        n_samples  = X.shape[0]
        indices    = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]

    # ── Get Feature Subset ───────────────────────────────────────────────────
    def _get_feature_indices(self, n_features):
        """
        Randomly select a subset of features for each tree
        This is what makes Random Forest different from just many trees
        """
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            k = min(self.max_features, n_features)
        else:
            k = n_features

        return np.random.choice(n_features, size=k, replace=False)

    # ── Train ─────────────────────────────────────────────────────────────────
    def fit(self, X, y):
        """Train n_trees decision trees on bootstrap samples"""
        X         = np.array(X)
        y         = np.array(y)
        self.trees = []

        print(f"  Training {self.n_trees} trees...")

        for i in range(self.n_trees):
            # Bootstrap sample
            X_sample, y_sample = self._bootstrap_sample(X, y)

            # Random feature subset
            feature_indices = self._get_feature_indices(X.shape[1])
            X_subset        = X_sample[:, feature_indices]

            # Train tree on subset
            tree = DecisionTree(
                max_depth         = self.max_depth,
                min_samples_split = self.min_samples_split,
                min_samples_leaf  = self.min_samples_leaf
            )
            tree.fit(X_subset, y_sample)

            self.trees.append((tree, feature_indices))

            if (i + 1) % 10 == 0:
                print(f"  Built {i+1} / {self.n_trees} trees...")

        print(f"All {self.n_trees} trees built!")
        return self

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, X):
        """
        Each tree votes, majority wins
        """
        X            = np.array(X)
        # Collect predictions from all trees
        all_preds    = np.zeros((len(self.trees), X.shape[0]))

        for i, (tree, feature_indices) in enumerate(self.trees):
            X_subset      = X[:, feature_indices]
            all_preds[i]  = tree.predict(X_subset)

        # Majority vote across all trees
        majority_votes = np.apply_along_axis(
            lambda x: np.bincount(x.astype(int)).argmax(),
            axis=0,
            arr=all_preds
        )
        return majority_votes

    def predict_proba(self, X):
        """
        Return probability of class 1
        = fraction of trees that voted for class 1
        """
        X         = np.array(X)
        all_preds = np.zeros((len(self.trees), X.shape[0]))

        for i, (tree, feature_indices) in enumerate(self.trees):
            X_subset     = X[:, feature_indices]
            all_preds[i] = tree.predict(X_subset)

        # Average votes = probability
        return all_preds.mean(axis=0)


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(os.path.join("src", "models"))

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    import time

    print("Testing Random Forest from scratch...")
    print("Loading dataset...")

    df = pd.read_csv(os.path.join("data", "processed", "final_dataset.csv"))

    # Use small sample for quick test
    df = df.sample(n=5000, random_state=42)

    feature_cols = [c for c in df.columns if c not in ["asn1", "asn2", "label"]]
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print("Training Random Forest (20 trees for quick test)...")

    start = time.time()
    rf    = RandomForest(
        n_trees           = 20,     # small for quick test
        max_depth         = 10,
        min_samples_split = 5,
        min_samples_leaf  = 1,
        max_features      = "sqrt"
    )
    rf.fit(X_train, y_train)
    train_time = time.time() - start

    print("Predicting...")
    start  = time.time()
    y_pred = rf.predict(X_test)
    pred_time = time.time() - start

    print("\n" + "="*40)
    print("RANDOM FOREST RESULTS")
    print("="*40)
    print(f"Overall Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Balanced Accuracy : {balanced_accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"F1 Score          : {f1_score(y_test, y_pred)*100:.2f}%")
    print(f"Training Time     : {train_time:.2f} sec")
    print(f"Prediction Time   : {pred_time:.2f} sec")
    print("="*40)
    print("Random Forest test complete!")
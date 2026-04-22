import numpy as np

class DecisionTree:
    """
    Decision Tree built from scratch.
    Supports binary classification using Gini impurity.
    """

    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1):
        """
        max_depth        : maximum depth of the tree
        min_samples_split: minimum samples required to split a node
        min_samples_leaf : minimum samples required at a leaf node
        """
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.root              = None

    # ── Gini Impurity ────────────────────────────────────────────────────────
    def _gini(self, y):
        """
        Gini impurity = 1 - sum(p_i^2)
        Lower is better (0 = pure node)
        """
        if len(y) == 0:
            return 0
        classes, counts = np.unique(y, return_counts=True)
        probs           = counts / len(y)
        return 1 - np.sum(probs ** 2)

    # ── Best Split ───────────────────────────────────────────────────────────
    def _best_split(self, X, y):
        """
        Find the best feature and threshold to split on
        by minimizing weighted Gini impurity
        """
        best_gini      = float("inf")
        best_feature   = None
        best_threshold = None

        n_samples, n_features = X.shape

        for feature in range(n_features):
            # Get unique thresholds for this feature
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                # Split data
                left_mask  = X[:, feature] <= threshold
                right_mask = ~left_mask

                y_left  = y[left_mask]
                y_right = y[right_mask]

                # Skip if split creates empty node or violates min_samples_leaf
                if len(y_left) < self.min_samples_leaf or \
                   len(y_right) < self.min_samples_leaf:
                    continue

                # Weighted gini
                n       = len(y)
                w_gini  = (len(y_left) / n)  * self._gini(y_left) + \
                          (len(y_right) / n) * self._gini(y_right)

                if w_gini < best_gini:
                    best_gini      = w_gini
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # ── Build Tree ───────────────────────────────────────────────────────────
    def _build_tree(self, X, y, depth=0):
        """Recursively build the decision tree"""
        n_samples = len(y)

        # ── Stopping conditions ──────────────────────────────────────────────
        # 1. Max depth reached
        if depth >= self.max_depth:
            return {"leaf": True, "value": self._leaf_value(y)}

        # 2. Too few samples to split
        if n_samples < self.min_samples_split:
            return {"leaf": True, "value": self._leaf_value(y)}

        # 3. Pure node (all same class)
        if len(np.unique(y)) == 1:
            return {"leaf": True, "value": y[0]}

        # ── Find best split ──────────────────────────────────────────────────
        best_feature, best_threshold = self._best_split(X, y)

        # No valid split found
        if best_feature is None:
            return {"leaf": True, "value": self._leaf_value(y)}

        # ── Split data ───────────────────────────────────────────────────────
        left_mask  = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_subtree  = self._build_tree(X[left_mask],  y[left_mask],  depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "leaf"      : False,
            "feature"   : best_feature,
            "threshold" : best_threshold,
            "left"      : left_subtree,
            "right"     : right_subtree,
        }

    def _leaf_value(self, y):
        """Return most common class in leaf"""
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    # ── Predict Single Sample ─────────────────────────────────────────────────
    def _predict_one(self, x, node):
        """Traverse tree for a single sample"""
        if node["leaf"]:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    # ── Public API ────────────────────────────────────────────────────────────
    def fit(self, X, y):
        """Train the decision tree"""
        X          = np.array(X)
        y          = np.array(y)
        self.root  = self._build_tree(X, y)
        return self

    def predict(self, X):
        """Predict class for each sample"""
        X = np.array(X)
        return np.array([self._predict_one(x, self.root) for x in X])

    def predict_proba(self, X):
        """
        Predict probability of class 1 for each sample
        Used by Random Forest and XGBoost
        """
        preds = self.predict(X)
        # Return as probability (0 or 1 for hard tree)
        return preds.astype(float)


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    print("Testing Decision Tree from scratch...")
    print("Loading dataset...")

    df = pd.read_csv(os.path.join("data", "processed", "final_dataset.csv"))

    # Use smaller sample for quick test
    df = df.sample(n=5000, random_state=42)

    feature_cols = [c for c in df.columns if c not in ["asn1", "asn2", "label"]]
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print("Training Decision Tree...")

    tree = DecisionTree(max_depth=10, min_samples_split=5)
    tree.fit(X_train, y_train)

    print("Predicting...")
    y_pred = tree.predict(X_test)

    print("\n" + "="*40)
    print("DECISION TREE RESULTS")
    print("="*40)
    print(f"Overall Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Balanced Accuracy : {balanced_accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"F1 Score          : {f1_score(y_test, y_pred)*100:.2f}%")
    print("="*40)
    print("✅ Decision Tree test complete!")

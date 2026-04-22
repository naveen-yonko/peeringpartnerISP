import numpy as np
from decision_tree import DecisionTree

class XGBoostScratch:
    """
    XGBoost (Extreme Gradient Boosting) built from scratch.
    
    Unlike Random Forest (parallel trees), XGBoost builds trees 
    SEQUENTIALLY — each tree corrects the errors of the previous ones.
    
    Key idea:
        prediction = base_score
                   + learning_rate * tree_1_output
                   + learning_rate * tree_2_output
                   + ...
    """
    
    def __init__(self, n_trees=100, max_depth=6, learning_rate=0.1,
                 min_samples_split=2, min_samples_leaf=1, subsample=0.8):
        """
        n_trees          : number of boosting rounds
        max_depth        : max depth of each tree
        learning_rate    : shrinks contribution of each tree (prevents overfitting)
        min_samples_split: min samples to split a node
        min_samples_leaf : min samples at leaf
        subsample        : fraction of samples used per tree (like dropout)
        """
        self.n_trees           = n_trees
        self.max_depth         = max_depth
        self.learning_rate     = learning_rate
        self.min_samples_split = min_samples_split
        self.min_samples_leaf  = min_samples_leaf
        self.subsample         = subsample
        self.trees             = []
        self.base_score        = 0.5   # initial prediction

    # ── Sigmoid ───────────────────────────────────────────────────────────────
    def _sigmoid(self, x):
        """Convert raw score to probability"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    # ── Gradients ─────────────────────────────────────────────────────────────
    def _compute_gradients(self, y, y_pred_proba):
        """
        Compute first order gradient (residuals) for binary cross entropy loss
        gradient = predicted_probability - actual_label
        """
        return y_pred_proba - y

    # ── Train ──────────────────────────────────────────────────────────────────
    def fit(self, X, y):
        """
        Train XGBoost:
        1. Start with base prediction
        2. Compute residuals (errors)
        3. Train a tree to predict residuals
        4. Update predictions
        5. Repeat n_trees times
        """
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n_samples = X.shape[0]

        # Start with base score
        raw_scores = np.full(n_samples, self.base_score)
        self.trees = []

        print(f"  Training {self.n_trees} boosting rounds...")

        for i in range(self.n_trees):

            # Current probabilities
            y_pred_proba = self._sigmoid(raw_scores)

            # Compute residuals (negative gradients)
            residuals = self._compute_gradients(y, y_pred_proba)

            # Subsample rows
            n_sub    = max(1, int(n_samples * self.subsample))
            indices  = np.random.choice(n_samples, size=n_sub, replace=False)
            X_sub    = X[indices]
            res_sub  = residuals[indices]

            # Train a tree on residuals
            tree = DecisionTree(
                max_depth         = self.max_depth,
                min_samples_split = self.min_samples_split,
                min_samples_leaf  = self.min_samples_leaf
            )
            tree.fit(X_sub, res_sub)
            self.trees.append(tree)

            # Update raw scores using learning rate
            update      = tree.predict(X)
            raw_scores -= self.learning_rate * update

            if (i + 1) % 10 == 0:
                y_pred       = (self._sigmoid(raw_scores) >= 0.5).astype(int)
                train_acc    = np.mean(y_pred == y) * 100
                print(f"  Round {i+1:3}/{self.n_trees} | Train Acc: {train_acc:.2f}%")

        print(f"All {self.n_trees} boosting rounds complete!")
        return self

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict_proba_raw(self, X):
        """Get raw probability scores"""
        X          = np.array(X, dtype=float)
        raw_scores = np.full(X.shape[0], self.base_score)

        for tree in self.trees:
            update      = tree.predict(X)
            raw_scores -= self.learning_rate * update

        return self._sigmoid(raw_scores)

    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        proba = self.predict_proba_raw(X)
        return (proba >= threshold).astype(int)

    def predict_proba(self, X):
        """Return probability of class 1"""
        return self.predict_proba_raw(X)


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

    print("Testing XGBoost from scratch...")
    print("Loading dataset...")

    df = pd.read_csv(os.path.join("data", "processed", "final_dataset.csv"))

    # Small sample for quick test
    df = df.sample(n=5000, random_state=42)

    feature_cols = [c for c in df.columns if c not in ["asn1", "asn2", "label"]]
    X = df[feature_cols].values
    y = df["label"].values

    # Normalize features (important for XGBoost)
    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    X       = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print("Training XGBoost (30 rounds for quick test)...")

    start = time.time()
    xgb   = XGBoostScratch(
        n_trees           = 30,
        max_depth         = 6,
        learning_rate     = 0.1,
        min_samples_split = 5,
        subsample         = 0.8
    )
    xgb.fit(X_train, y_train)
    train_time = time.time() - start

    print("Predicting...")
    start  = time.time()
    y_pred = xgb.predict(X_test)
    pred_time = time.time() - start

    print("\n" + "="*40)
    print("XGBOOST RESULTS")
    print("="*40)
    print(f"Overall Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Balanced Accuracy : {balanced_accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"F1 Score          : {f1_score(y_test, y_pred)*100:.2f}%")
    print(f"Training Time     : {train_time:.2f} sec")
    print(f"Prediction Time   : {pred_time:.2f} sec")
    print("="*40)
    print("XGBoost test complete!")
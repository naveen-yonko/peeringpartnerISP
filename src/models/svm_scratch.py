import numpy as np

class SVMScratch:
    """
    Support Vector Machine built from scratch.
    Uses gradient descent to optimize the hinge loss.
    
    Key idea:
        Find a hyperplane that maximizes the margin between classes.
        
        Decision: sign(w·x + b)
        Loss    : hinge loss + regularization
                  L = lambda*||w||^2 + mean(max(0, 1 - y*(w·x + b)))
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01,
                 n_epochs=100, batch_size=256):
        """
        learning_rate : step size for gradient descent
        lambda_param  : regularization strength (prevents overfitting)
        n_epochs      : number of passes through the data
        batch_size    : mini-batch size for stochastic gradient descent
        """
        self.learning_rate = learning_rate
        self.lambda_param  = lambda_param
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.w             = None   # weights
        self.b             = 0.0   # bias

    # ── Convert labels to -1 / +1 ────────────────────────────────────────────
    def _convert_labels(self, y):
        """SVM needs labels as -1 and +1 (not 0 and 1)"""
        return np.where(y <= 0, -1, 1).astype(float)

    # ── Hinge Loss ────────────────────────────────────────────────────────────
    def _hinge_loss(self, X, y_svm):
        """
        Hinge loss = max(0, 1 - y*(w·x + b))
        """
        margins = y_svm * (X @ self.w + self.b)
        loss    = np.maximum(0, 1 - margins)
        return np.mean(loss) + self.lambda_param * np.dot(self.w, self.w)

    # ── Gradients ─────────────────────────────────────────────────────────────
    def _compute_gradients(self, X_batch, y_batch):
        """
        Compute gradients of hinge loss w.r.t w and b
        
        If y*(w·x + b) >= 1: correct side, no loss
            dw = 2*lambda*w
            db = 0
        Else: wrong side or inside margin
            dw = 2*lambda*w - y*x
            db = -y
        """
        margins  = y_batch * (X_batch @ self.w + self.b)
        mask     = margins >= 1   # correctly classified with margin

        # Gradient w.r.t weights
        dw = 2 * self.lambda_param * self.w
        db = 0.0

        # For misclassified or margin violations
        wrong = ~mask
        if wrong.any():
            dw -= np.mean(y_batch[wrong, np.newaxis] * X_batch[wrong], axis=0)
            db -= np.mean(y_batch[wrong])

        return dw, db

    # ── Train ──────────────────────────────────────────────────────────────────
    def fit(self, X, y):
        """
        Train SVM using mini-batch gradient descent
        """
        X       = np.array(X, dtype=float)
        y_svm   = self._convert_labels(y)

        n_samples, n_features = X.shape

        # Initialize weights to zeros
        self.w = np.zeros(n_features)
        self.b = 0.0

        print(f"Training SVM for {self.n_epochs} epochs...")

        for epoch in range(self.n_epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(n_samples)
            X_shuf  = X[indices]
            y_shuf  = y_svm[indices]

            # Mini-batch gradient descent
            for start in range(0, n_samples, self.batch_size):
                end     = min(start + self.batch_size, n_samples)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]

                dw, db = self._compute_gradients(X_batch, y_batch)

                # Update weights
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                loss     = self._hinge_loss(X, y_svm)
                y_pred   = self.predict(X)
                train_acc = np.mean(y_pred == y) * 100
                print(f"  Epoch {epoch+1:3}/{self.n_epochs} | "
                      f"Loss: {loss:.4f} | Train Acc: {train_acc:.2f}%")

        print(f"Training complete!")
        return self

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, X):
        """
        Predict class labels (0 or 1)
        Decision: sign(w·x + b)
        """
        X         = np.array(X, dtype=float)
        raw       = X @ self.w + self.b
        # Convert back from -1/+1 to 0/1
        return np.where(raw >= 0, 1, 0).astype(int)

    def predict_proba(self, X):
        """
        Return a soft probability using sigmoid on raw scores
        """
        X   = np.array(X, dtype=float)
        raw = X @ self.w + self.b
        return 1 / (1 + np.exp(-np.clip(raw, -500, 500)))


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(os.path.join("src", "models"))

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    import time

    print("Testing SVM from scratch...")
    print("Loading dataset...")

    df = pd.read_csv(os.path.join("data", "processed", "final_dataset.csv"))

    # Small sample for quick test
    df = df.sample(n=5000, random_state=42)

    feature_cols = [c for c in df.columns if c not in ["asn1", "asn2", "label"]]
    X = df[feature_cols].values
    y = df["label"].values

    # Normalize — very important for SVM!
    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print("Training SVM...")

    start = time.time()
    svm   = SVMScratch(
        learning_rate = 0.001,
        lambda_param  = 0.01,
        n_epochs      = 100,
        batch_size    = 256
    )
    svm.fit(X_train, y_train)
    train_time = time.time() - start

    print("Predicting...")
    start     = time.time()
    y_pred    = svm.predict(X_test)
    pred_time = time.time() - start

    print("\n" + "="*40)
    print("SVM RESULTS")
    print("="*40)
    print(f"Overall Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Balanced Accuracy : {balanced_accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"F1 Score          : {f1_score(y_test, y_pred)*100:.2f}%")
    print(f"Training Time     : {train_time:.2f} sec")
    print(f"Prediction Time   : {pred_time:.2f} sec")
    print("="*40)
    print("SVM test complete!")
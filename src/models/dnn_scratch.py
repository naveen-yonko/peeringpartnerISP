import numpy as np

class DNNScratch:
    """
    Deep Neural Network built from scratch.
    Architecture: Input -> Hidden1 -> Hidden2 -> Hidden3 -> Output
    
    Uses:
    - ReLU activation for hidden layers
    - Sigmoid activation for output layer
    - Binary Cross Entropy loss
    - Mini-batch gradient descent
    """

    def __init__(self, hidden_layers=[100, 100, 100],
                 learning_rate=0.001, n_epochs=100,
                 batch_size=256, random_state=42):
        """
        hidden_layers : list of neurons per hidden layer
                        e.g. [100, 100, 100] = 3 hidden layers, 100 neurons each
        learning_rate : step size for gradient descent
        n_epochs      : number of passes through data
        batch_size    : mini-batch size
        random_state  : for reproducibility
        """
        self.hidden_layers  = hidden_layers
        self.learning_rate  = learning_rate
        self.n_epochs       = n_epochs
        self.batch_size     = batch_size
        self.random_state   = random_state
        self.weights        = []   # W matrices
        self.biases         = []   # b vectors

    # ── Weight Initialization ─────────────────────────────────────────────────
    def _initialize_weights(self, n_features):
        """
        He initialization — works well with ReLU
        W ~ N(0, sqrt(2/n_inputs))
        """
        np.random.seed(self.random_state)
        self.weights = []
        self.biases  = []

        # Build layer sizes
        # e.g. [36, 100, 100, 100, 1]
        layer_sizes = [n_features] + self.hidden_layers + [1]

        for i in range(len(layer_sizes) - 1):
            n_in  = layer_sizes[i]
            n_out = layer_sizes[i + 1]

            # He initialization
            W = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            b = np.zeros((1, n_out))

            self.weights.append(W)
            self.biases.append(b)

    # ── Activation Functions ──────────────────────────────────────────────────
    def _relu(self, Z):
        """ReLU: max(0, Z)"""
        return np.maximum(0, Z)

    def _relu_derivative(self, Z):
        """Derivative of ReLU: 1 if Z > 0 else 0"""
        return (Z > 0).astype(float)

    def _sigmoid(self, Z):
        """Sigmoid: 1 / (1 + e^-Z)"""
        return 1 / (1 + np.exp(-np.clip(Z, -500, 500)))

    # ── Forward Pass ──────────────────────────────────────────────────────────
    def _forward(self, X):
        """
        Pass input through all layers
        Returns activations and pre-activations for backprop
        """
        activations     = [X]   # A0 = input
        pre_activations = []    # Z values before activation

        A = X
        for i, (W, b) in enumerate(zip(self.weights, self.biases)):
            Z = A @ W + b           # linear transformation
            pre_activations.append(Z)

            # Last layer uses sigmoid, hidden layers use ReLU
            if i == len(self.weights) - 1:
                A = self._sigmoid(Z)
            else:
                A = self._relu(Z)

            activations.append(A)

        return activations, pre_activations

    # ── Loss ──────────────────────────────────────────────────────────────────
    def _binary_cross_entropy(self, y, y_pred):
        """
        BCE Loss = -mean(y*log(p) + (1-y)*log(1-p))
        """
        epsilon = 1e-15   # prevent log(0)
        y_pred  = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    # ── Backward Pass ─────────────────────────────────────────────────────────
    def _backward(self, X, y, activations, pre_activations):
        """
        Backpropagation — compute gradients layer by layer
        going backwards from output to input
        """
        n          = X.shape[0]
        n_layers   = len(self.weights)
        dW_list    = [None] * n_layers
        db_list    = [None] * n_layers

        # Output layer gradient
        # dL/dA_last = -(y/p - (1-y)/(1-p))
        # Combined with sigmoid derivative: dL/dZ_last = p - y
        A_last = activations[-1]
        y      = y.reshape(-1, 1)
        dZ     = A_last - y          # shape: (n, 1)

        # Backpropagate through each layer
        for i in reversed(range(n_layers)):
            A_prev = activations[i]       # input to this layer

            # Gradients for weights and biases
            dW_list[i] = (A_prev.T @ dZ) / n
            db_list[i] = np.mean(dZ, axis=0, keepdims=True)

            # Gradient for previous layer (stop at input layer)
            if i > 0:
                dA_prev = dZ @ self.weights[i].T
                dZ      = dA_prev * self._relu_derivative(pre_activations[i-1])

        return dW_list, db_list

    # ── Update Weights ────────────────────────────────────────────────────────
    def _update_weights(self, dW_list, db_list):
        """Gradient descent weight update"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dW_list[i]
            self.biases[i]  -= self.learning_rate * db_list[i]

    # ── Train ──────────────────────────────────────────────────────────────────
    def fit(self, X, y):
        """Train DNN using mini-batch gradient descent"""
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        n_samples, n_features = X.shape
        self._initialize_weights(n_features)

        print(f"  Architecture: {n_features} → "
              f"{' → '.join(map(str, self.hidden_layers))} → 1")
        print(f"  Training for {self.n_epochs} epochs...")

        for epoch in range(self.n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuf  = X[indices]
            y_shuf  = y[indices]

            epoch_loss = 0
            n_batches  = 0

            # Mini-batch gradient descent
            for start in range(0, n_samples, self.batch_size):
                end     = min(start + self.batch_size, n_samples)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]

                # Forward pass
                activations, pre_activations = self._forward(X_batch)

                # Loss
                y_pred     = activations[-1].flatten()
                batch_loss = self._binary_cross_entropy(y_batch, y_pred)
                epoch_loss += batch_loss
                n_batches  += 1

                # Backward pass
                dW_list, db_list = self._backward(
                    X_batch, y_batch, activations, pre_activations
                )

                # Update weights
                self._update_weights(dW_list, db_list)

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                avg_loss  = epoch_loss / n_batches
                y_pred    = self.predict(X)
                train_acc = np.mean(y_pred == y) * 100
                print(f"  Epoch {epoch+1:3}/{self.n_epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}%")

        print(f"Training complete!")
        return self

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        X              = np.array(X, dtype=float)
        activations, _ = self._forward(X)
        proba          = activations[-1].flatten()
        return (proba >= threshold).astype(int)

    def predict_proba(self, X):
        """Return probability of class 1"""
        X              = np.array(X, dtype=float)
        activations, _ = self._forward(X)
        return activations[-1].flatten()


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

    print("Testing DNN from scratch...")
    print("Loading dataset...")

    df = pd.read_csv(os.path.join("data", "processed", "final_dataset.csv"))

    # Small sample for quick test
    df = df.sample(n=5000, random_state=42)

    feature_cols = [c for c in df.columns if c not in ["asn1", "asn2", "label"]]
    X = df[feature_cols].values
    y = df["label"].values

    # Normalize — important for DNN!
    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print("Training DNN [Input→100→100→100→Output]...")

    start = time.time()
    dnn   = DNNScratch(
        hidden_layers = [100, 100, 100],
        learning_rate = 0.001,
        n_epochs      = 100,
        batch_size    = 256
    )
    dnn.fit(X_train, y_train)
    train_time = time.time() - start

    print("Predicting...")
    start     = time.time()
    y_pred    = dnn.predict(X_test)
    pred_time = time.time() - start

    print("\n" + "="*40)
    print("DNN RESULTS")    
    print("="*40)
    print(f"Overall Accuracy  : {accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Balanced Accuracy : {balanced_accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"F1 Score          : {f1_score(y_test, y_pred)*100:.2f}%")
    print(f"Training Time     : {train_time:.2f} sec")
    print(f"Prediction Time   : {pred_time:.4f} sec")
    print("="*40)
    print("DNN test complete!")
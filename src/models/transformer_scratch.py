import numpy as np

class TransformerScratch:
    """
    Simplified Transformer built from scratch for tabular data.
    Uses fully analytical gradients — much faster than numerical gradients.

    Architecture:
        Input (batch, n_features)
             ↓
        Linear Embedding → (batch, n_features, d_model)
             ↓
        Self Attention + Add & Norm
             ↓
        Feed Forward + Add & Norm
             ↓
        Global Average Pooling → (batch, d_model)
             ↓
        Classification Head → sigmoid → prediction
    """

    def __init__(self, d_model=16, n_heads=2, d_ff=32,
                 learning_rate=0.001, n_epochs=50,
                 batch_size=256, random_state=42):
        self.d_model       = d_model
        self.n_heads       = n_heads
        self.d_ff          = d_ff
        self.learning_rate = learning_rate
        self.n_epochs      = n_epochs
        self.batch_size    = batch_size
        self.random_state  = random_state
        self.params        = {}

    # ── Initialize Parameters ─────────────────────────────────────────────────
    def _init_params(self, n_features):
        np.random.seed(self.random_state)
        d  = self.d_model
        p  = {}

        # Embedding layer
        p["W_embed"] = np.random.randn(n_features, d) * 0.01
        p["b_embed"] = np.zeros((1, n_features, d))

        # Attention weights
        p["W_Q"] = np.random.randn(d, d) * 0.01
        p["W_K"] = np.random.randn(d, d) * 0.01
        p["W_V"] = np.random.randn(d, d) * 0.01
        p["W_O"] = np.random.randn(d, d) * 0.01

        # Feed forward
        p["W_ff1"] = np.random.randn(d, self.d_ff) * 0.01
        p["b_ff1"] = np.zeros((1, 1, self.d_ff))
        p["W_ff2"] = np.random.randn(self.d_ff, d) * 0.01
        p["b_ff2"] = np.zeros((1, 1, d))

        # Layer norm
        p["ln1_g"] = np.ones((1, 1, d))
        p["ln1_b"] = np.zeros((1, 1, d))
        p["ln2_g"] = np.ones((1, 1, d))
        p["ln2_b"] = np.zeros((1, 1, d))

        # Classification head
        p["W_cls"] = np.random.randn(d, 1) * 0.01
        p["b_cls"] = np.zeros(1)

        self.params     = p
        self.n_features = n_features

    # ── Activations ───────────────────────────────────────────────────────────
    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_grad(self, x):
        return (x > 0).astype(float)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def _softmax(self, x):
        x  = x - x.max(axis=-1, keepdims=True)
        ex = np.exp(x)
        return ex / ex.sum(axis=-1, keepdims=True)

    # ── Layer Norm ────────────────────────────────────────────────────────────
    def _layer_norm(self, x, g, b, eps=1e-6):
        self._ln_x   = x
        self._ln_mean = x.mean(axis=-1, keepdims=True)
        self._ln_std  = x.std(axis=-1, keepdims=True) + eps
        x_hat        = (x - self._ln_mean) / self._ln_std
        self._ln_xhat = x_hat
        return g * x_hat + b

    def _layer_norm_grad(self, dout, g, eps=1e-6):
        """Analytical gradient of layer norm"""
        N    = dout.shape[-1]
        xhat = self._ln_xhat
        std  = self._ln_std
        dg   = (dout * xhat).sum(axis=(0, 1), keepdims=True)
        db   = dout.sum(axis=(0, 1), keepdims=True)
        dxhat = dout * g
        dx   = (1/std) * (dxhat - dxhat.mean(axis=-1, keepdims=True)
                - xhat * (dxhat * xhat).mean(axis=-1, keepdims=True))
        return dx, dg, db

    # ── Self Attention ────────────────────────────────────────────────────────
    def _attention_forward(self, H):
        """
        Scaled dot product self attention
        H: (batch, seq, d)
        """
        p     = self.params
        scale = np.sqrt(self.d_model)

        Q = H @ p["W_Q"]   # (batch, seq, d)
        K = H @ p["W_K"]
        V = H @ p["W_V"]

        # Attention scores
        scores = Q @ K.transpose(0, 2, 1) / scale  # (batch, seq, seq)
        attn   = self._softmax(scores)              # (batch, seq, seq)
        out    = attn @ V                           # (batch, seq, d)
        out    = out @ p["W_O"]                     # (batch, seq, d)

        # Cache for backward
        self._attn_H     = H
        self._attn_Q     = Q
        self._attn_K     = K
        self._attn_V     = V
        self._attn_scores = scores
        self._attn_probs  = attn
        self._attn_out    = out

        return out

    def _attention_backward(self, dout):
        """Analytical gradients for attention"""
        p     = self.params
        scale = np.sqrt(self.d_model)
        attn  = self._attn_probs
        H     = self._attn_H
        Q     = self._attn_Q
        K     = self._attn_K
        V     = self._attn_V

        # Gradient through W_O
        dW_O  = self._attn_out.transpose(0, 2, 1) @ dout
        dW_O  = dW_O.sum(axis=0)
        d_av  = dout @ p["W_O"].T           # (batch, seq, d)

        # Gradient through V
        dV    = attn.transpose(0, 2, 1) @ d_av
        d_attn = d_av @ V.transpose(0, 2, 1)

        # Gradient through softmax
        # d_scores = attn * (d_attn - (d_attn * attn).sum(-1, keepdims=True))
        sa     = (d_attn * attn).sum(axis=-1, keepdims=True)
        d_scores = attn * (d_attn - sa) / scale

        # Gradient through Q, K
        dQ    = d_scores @ K
        dK    = d_scores.transpose(0, 2, 1) @ Q

        dW_Q  = H.transpose(0, 2, 1) @ dQ
        dW_Q  = dW_Q.sum(axis=0)
        dW_K  = H.transpose(0, 2, 1) @ dK
        dW_K  = dW_K.sum(axis=0)
        dW_V  = H.transpose(0, 2, 1) @ dV
        dW_V  = dW_V.sum(axis=0)

        # Gradient w.r.t input H
        dH = (dQ @ p["W_Q"].T +
              dK @ p["W_K"].T +
              dV @ p["W_V"].T)

        return dH, dW_Q, dW_K, dW_V, dW_O

    # ── Forward Pass ──────────────────────────────────────────────────────────
    def _forward(self, X):
        """Full forward pass — cache everything needed for backward"""
        p = self.params

        # 1. Embedding: each feature gets its own d_model vector
        # X: (batch, n_features) → (batch, n_features, d_model)
        # Use diagonal of W_embed as per-feature scalar embedding
        H = X[:, :, np.newaxis] * p["W_embed"][np.newaxis, :, :] \
            + p["b_embed"]

        self._H0 = X   # cache input

        # 2. Self Attention
        attn_out     = self._attention_forward(H)
        H_res1       = H + attn_out          # residual connection
        self._H_res1 = H_res1

        # Layer norm 1
        H_ln1        = self._layer_norm(H_res1, p["ln1_g"], p["ln1_b"])
        self._H_ln1  = H_ln1
        self._ln1_mean = self._ln_mean
        self._ln1_std  = self._ln_std
        self._ln1_xhat = self._ln_xhat

        # 3. Feed Forward
        ff1          = self._relu(H_ln1 @ p["W_ff1"] + p["b_ff1"])
        self._ff1    = ff1
        self._ff1_pre = H_ln1 @ p["W_ff1"] + p["b_ff1"]
        ff2          = ff1 @ p["W_ff2"] + p["b_ff2"]
        H_res2       = H_ln1 + ff2          # residual connection
        self._H_res2 = H_res2

        # Layer norm 2
        H_ln2        = self._layer_norm(H_res2, p["ln2_g"], p["ln2_b"])
        self._H_ln2  = H_ln2
        self._ln2_mean = self._ln_mean
        self._ln2_std  = self._ln_std
        self._ln2_xhat = self._ln_xhat

        # 4. Global average pool
        H_pool       = H_ln2.mean(axis=1)   # (batch, d_model)
        self._H_pool = H_pool

        # 5. Classification head
        logits = H_pool @ p["W_cls"] + p["b_cls"]
        proba  = self._sigmoid(logits).flatten()

        return proba

    # ── Backward Pass ─────────────────────────────────────────────────────────
    def _backward(self, X, y, proba):
        """Full analytical backward pass"""
        p     = self.params
        batch = X.shape[0]
        grads = {}

        # 1. Classification head gradient
        delta         = (proba - y).reshape(-1, 1) / batch
        grads["W_cls"] = self._H_pool.T @ delta
        grads["b_cls"] = delta.sum(axis=0)

        # Gradient into pooling
        dH_pool = delta @ p["W_cls"].T          # (batch, d)
        dH_ln2  = np.repeat(
            dH_pool[:, np.newaxis, :],
            self.n_features, axis=1
        ) / self.n_features                      # (batch, seq, d)

        # 2. Layer norm 2 gradient
        self._ln_xhat = self._ln2_xhat
        self._ln_std  = self._ln2_std
        dH_res2, grads["ln2_g"], grads["ln2_b"] = \
            self._layer_norm_grad(dH_ln2, p["ln2_g"])

        # 3. Feed forward gradient
        dff2          = dH_res2
        grads["W_ff2"] = self._ff1.transpose(0, 2, 1) @ dff2
        grads["W_ff2"] = grads["W_ff2"].sum(axis=0)
        grads["b_ff2"] = dff2.sum(axis=(0, 1), keepdims=True)
        dff1           = dff2 @ p["W_ff2"].T
        dff1_pre       = dff1 * self._relu_grad(self._ff1_pre)
        grads["W_ff1"] = self._H_ln1.transpose(0, 2, 1) @ dff1_pre
        grads["W_ff1"] = grads["W_ff1"].sum(axis=0)
        grads["b_ff1"] = dff1_pre.sum(axis=(0, 1), keepdims=True)

        dH_ln1_ff = dff1_pre @ p["W_ff1"].T
        dH_ln1    = dH_res2 + dH_ln1_ff   # residual

        # 4. Layer norm 1 gradient
        self._ln_xhat = self._ln1_xhat
        self._ln_std  = self._ln1_std
        dH_res1, grads["ln1_g"], grads["ln1_b"] = \
            self._layer_norm_grad(dH_ln1, p["ln1_g"])

        # 5. Attention gradient
        dH_attn, grads["W_Q"], grads["W_K"], \
            grads["W_V"], grads["W_O"] = \
            self._attention_backward(dH_res1)

        dH = dH_res1 + dH_attn   # residual

        # 6. Embedding gradient
        grads["W_embed"] = (
            self._H0[:, :, np.newaxis] * dH
        ).sum(axis=0)
        grads["b_embed"] = dH.sum(axis=0, keepdims=True)

        return grads

    # ── Update ────────────────────────────────────────────────────────────────
    def _update(self, grads):
        for key in self.params:
            if key in grads:
                self.params[key] -= self.learning_rate * grads[key]

    # ── Train ──────────────────────────────────────────────────────────────────
    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        n_samples, n_features = X.shape
        self._init_params(n_features)

        print(f"  Architecture: {n_features} → "
              f"Embed(d={self.d_model}) → "
              f"Attention(h={self.n_heads}) → FFN → Pool → Sigmoid")
        print(f"  Training for {self.n_epochs} epochs...")

        for epoch in range(self.n_epochs):
            idx    = np.random.permutation(n_samples)
            X_shuf = X[idx]
            y_shuf = y[idx]

            epoch_loss = 0
            n_batches  = 0

            for start in range(0, n_samples, self.batch_size):
                end     = min(start + self.batch_size, n_samples)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]

                # Forward
                proba      = self._forward(X_batch)
                eps        = 1e-15
                proba_clip = np.clip(proba, eps, 1 - eps)
                loss       = -np.mean(
                    y_batch * np.log(proba_clip) +
                    (1 - y_batch) * np.log(1 - proba_clip)
                )
                epoch_loss += loss
                n_batches  += 1

                # Backward
                grads = self._backward(X_batch, y_batch, proba)
                self._update(grads)

            if (epoch + 1) % 10 == 0:
                proba_all = self._forward(X)
                train_acc = np.mean((proba_all >= 0.5) == y) * 100
                avg_loss  = epoch_loss / n_batches
                print(f"  Epoch {epoch+1:3}/{self.n_epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Train Acc: {train_acc:.2f}%")

        print("  ✅ Training complete!")
        return self

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, X, threshold=0.5):
        X     = np.array(X, dtype=float)
        proba = self._forward(X)
        return (proba >= threshold).astype(int)

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        return self._forward(X)


# ── Quick Test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    import sys
    os.chdir(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    sys.path.append(os.path.join("src", "models"))

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score,
                                 balanced_accuracy_score, f1_score)
    import time

    print("Testing Transformer from scratch...")
    print("Loading dataset...")

    df = pd.read_csv(
        os.path.join("data", "processed", "final_dataset.csv"))

    df = df.sample(n=3000, random_state=42)

    feature_cols = [c for c in df.columns
                    if c not in ["asn1", "asn2", "label"]]
    X = df[feature_cols].values
    y = df["label"].values

    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print("Training Transformer...")

    start = time.time()
    xfmr  = TransformerScratch(
        d_model       = 16,
        n_heads       = 2,
        d_ff          = 32,
        learning_rate = 0.001,
        n_epochs      = 50,
        batch_size    = 256
    )
    xfmr.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred    = xfmr.predict(X_test)
    pred_time = time.time() - start - train_time

    print("\n" + "="*40)
    print("TRANSFORMER RESULTS")
    print("="*40)
    print(f"Overall Accuracy  : "
          f"{accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"Balanced Accuracy : "
          f"{balanced_accuracy_score(y_test, y_pred)*100:.2f}%")
    print(f"F1 Score          : "
          f"{f1_score(y_test, y_pred)*100:.2f}%")
    print(f"Training Time     : {train_time:.2f} sec")
    print(f"Prediction Time   : {pred_time:.4f} sec")
    print("="*40)
    print("Transformer test complete!")
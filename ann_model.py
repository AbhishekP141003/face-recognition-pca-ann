"""
ann_model.py
------------
Artificial Neural Network (MLP) built completely from scratch using NumPy.

Architecture:
    Input  (k features from PCA)
      -> Hidden Layer 1  (128 neurons, ReLU)
      -> Hidden Layer 2  (64  neurons, ReLU)
      -> Output Layer    (n_classes neurons, Softmax)

Training:
    Mini-batch Stochastic Gradient Descent (SGD)
    Backpropagation
    Cross-Entropy Loss

Saving/Loading:
    Model weights saved as .npz file using NumPy.
"""

import os
import numpy as np
from sklearn.metrics import accuracy_score


# =============================================================================
# Activation Functions
# =============================================================================

def relu(z):
    """
    ReLU activation: f(z) = max(0, z)
    Used in hidden layers. Adds non-linearity without vanishing gradient.
    """
    return np.maximum(0, z)

def relu_grad(z):
    """Derivative of ReLU: 1 if z > 0 else 0."""
    return (z > 0).astype(np.float64)

def softmax(z):
    """
    Softmax activation for output layer.
    Converts raw scores to class probabilities (sum = 1).

    Formula: softmax(z_i) = exp(z_i) / sum(exp(z_j))

    Numerical stability trick: subtract max before exp.
    """
    z_stable = z - z.max(axis=1, keepdims=True)
    e        = np.exp(z_stable)
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(probs, y_onehot):
    """
    Cross-entropy loss for multi-class classification.

    Formula: L = -(1/m) * sum(y_true * log(y_pred))
    """
    return -np.mean(np.sum(y_onehot * np.log(probs + 1e-12), axis=1))


# =============================================================================
# ANN Class
# =============================================================================

class ANN:
    """
    Multi-Layer Perceptron (MLP) for face classification.

    Parameters
    ----------
    layer_sizes  : list  — hidden layer neuron counts e.g. [128, 64]
    n_classes    : int   — number of output classes
    lr           : float — learning rate for SGD
    epochs       : int   — number of training epochs
    batch_size   : int   — mini-batch size
    random_state : int   — seed for reproducibility
    """

    def __init__(self, layer_sizes, n_classes,
                 lr=0.005, epochs=300, batch_size=16, random_state=42):
        self.layer_sizes   = layer_sizes
        self.n_classes     = n_classes
        self.lr            = lr
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.rng           = np.random.default_rng(random_state)
        self.weights_      = []
        self.biases_       = []
        self.loss_history_ = []

    # -------------------------------------------------------------------------
    # Weight Initialisation — He Initialisation
    # Formula: W ~ Normal(0, sqrt(2 / fan_in))
    # Better than random init for ReLU networks.
    # -------------------------------------------------------------------------

    def _init_params(self, n_input):
        dims = [n_input] + self.layer_sizes + [self.n_classes]
        self.weights_ = []
        self.biases_  = []
        for i in range(len(dims) - 1):
            fan_in = dims[i]
            W = self.rng.standard_normal((dims[i], dims[i+1])) \
                * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, dims[i+1]))
            self.weights_.append(W)
            self.biases_.append(b)

    # -------------------------------------------------------------------------
    # Forward Pass
    # For each layer l:
    #   Z[l] = A[l-1] * W[l] + b[l]
    #   A[l] = ReLU(Z[l])          for hidden layers
    #   A[L] = Softmax(Z[L])       for output layer
    # -------------------------------------------------------------------------

    def _forward(self, X):
        """
        Forward propagation through all layers.

        Returns
        -------
        activations : list of A[0]..A[L]  (A[0] = input X)
        pre_acts    : list of Z[1]..Z[L]
        """
        activations = [X]
        pre_acts    = []
        A = X
        for l, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            Z = A @ W + b
            pre_acts.append(Z)
            if l < len(self.weights_) - 1:
                A = relu(Z)        # hidden layer
            else:
                A = softmax(Z)     # output layer
            activations.append(A)
        return activations, pre_acts

    # -------------------------------------------------------------------------
    # Backward Pass (Backpropagation)
    # Compute gradients using chain rule:
    #
    # Output layer delta:
    #   dZ[L] = A[L] - Y           (softmax + cross-entropy combined)
    #
    # Hidden layer delta:
    #   dZ[l] = (dZ[l+1] * W[l+1]^T) * relu'(Z[l])
    #
    # Weight gradients:
    #   dW[l] = (1/m) * A[l-1]^T * dZ[l]
    #   db[l] = (1/m) * sum(dZ[l])
    # -------------------------------------------------------------------------

    def _backward(self, activations, pre_acts, y_onehot):
        m        = y_onehot.shape[0]
        dW_list  = []
        db_list  = []

        # Output layer gradient
        dZ = activations[-1] - y_onehot       # (m, n_classes)

        for l in range(len(self.weights_) - 1, -1, -1):
            dW = (activations[l].T @ dZ) / m
            db = dZ.mean(axis=0, keepdims=True)
            dW_list.insert(0, dW)
            db_list.insert(0, db)

            if l > 0:
                dA = dZ @ self.weights_[l].T
                dZ = dA * relu_grad(pre_acts[l - 1])

        # SGD parameter update
        for l in range(len(self.weights_)):
            self.weights_[l] -= self.lr * dW_list[l]
            self.biases_[l]  -= self.lr * db_list[l]

    # -------------------------------------------------------------------------
    # One-Hot Encode Labels
    # -------------------------------------------------------------------------

    def _onehot(self, y):
        Y = np.zeros((len(y), self.n_classes))
        Y[np.arange(len(y)), y] = 1
        return Y

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------

    def fit(self, X, y):
        """
        Train the ANN using mini-batch SGD.

        Parameters
        ----------
        X : ndarray (n_samples, k)   — PCA signatures (transposed)
        y : ndarray (n_samples,)     — integer class labels (0-based)
        """
        self._init_params(X.shape[1])
        Y_oh = self._onehot(y)
        N    = X.shape[0]

        print(f"\n--- ANN TRAINING ---")
        print(f"Architecture : {X.shape[1]} -> "
              f"{self.layer_sizes} -> {self.n_classes}")
        print(f"Epochs: {self.epochs}  |  LR: {self.lr}  |  "
              f"Batch: {self.batch_size}\n")

        for epoch in range(1, self.epochs + 1):
            # Shuffle data each epoch
            perm   = self.rng.permutation(N)
            X_s, Y_s = X[perm], Y_oh[perm]

            # Mini-batch updates
            for start in range(0, N, self.batch_size):
                Xb = X_s[start:start + self.batch_size]
                Yb = Y_s[start:start + self.batch_size]
                acts, pre = self._forward(Xb)
                self._backward(acts, pre, Yb)

            # Record loss on full training set
            acts, _ = self._forward(X)
            loss    = cross_entropy_loss(acts[-1], Y_oh)
            self.loss_history_.append(loss)

            if epoch % 50 == 0 or epoch == 1:
                acc = accuracy_score(y, np.argmax(acts[-1], axis=1))
                print(f"  Epoch {epoch:4d}/{self.epochs}  "
                      f"loss={loss:.4f}  train_acc={acc:.3f}")

        return self

    # -------------------------------------------------------------------------
    # Prediction
    # -------------------------------------------------------------------------

    def predict(self, X):
        """
        Predict class labels for input X.

        Parameters
        ----------
        X : ndarray (n_samples, k)

        Returns
        -------
        y_pred : ndarray (n_samples,)
        """
        acts, _ = self._forward(X)
        return np.argmax(acts[-1], axis=1)

    def predict_proba(self, X):
        """Return class probabilities."""
        acts, _ = self._forward(X)
        return acts[-1]

    # -------------------------------------------------------------------------
    # Save and Load Model
    # -------------------------------------------------------------------------

    def save(self, path="outputs/ann_model.npz"):
        """Save model weights and biases to .npz file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_dict = {}
        for i, (W, b) in enumerate(zip(self.weights_, self.biases_)):
            save_dict[f"W{i}"] = W
            save_dict[f"b{i}"] = b
        save_dict["n_layers"]   = np.array([len(self.weights_)])
        save_dict["n_classes"]  = np.array([self.n_classes])
        save_dict["layer_sizes"]= np.array(self.layer_sizes)
        np.savez(path, **save_dict)
        print(f"[MODEL] ANN saved -> {path}")

    def load(self, path="outputs/ann_model.npz"):
        """Load model weights and biases from .npz file."""
        data = np.load(path, allow_pickle=True)
        n_layers = int(data["n_layers"][0])
        self.weights_ = [data[f"W{i}"] for i in range(n_layers)]
        self.biases_  = [data[f"b{i}"] for i in range(n_layers)]
        print(f"[MODEL] ANN loaded from {path}")
        return self
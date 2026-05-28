"""
train.py
--------
Training pipeline for Face Recognition using PCA + ANN.

What this file does:
  1. Loads dataset
  2. Splits into 60% train / 40% test
  3. Runs all PCA steps (Steps 2-8)
  4. Normalises PCA signatures
  5. Trains ANN on PCA features
  6. Saves PCA components and ANN model
  7. Generates all training visualizations
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split

from utils     import (load_dataset, show_mean_face, show_eigenfaces,
                       show_training_curve, show_k_vs_accuracy)
from pca       import PCA
from ann_model import ANN


# =============================================================================
# CONFIGURATION — change these values to experiment
# =============================================================================
DATASET_DIR  = "dataset/faces"
N_COMPONENTS = 50          # k: number of eigenfaces
HIDDEN_LAYERS= [128, 64]   # ANN hidden layer sizes
LR           = 0.005       # learning rate
EPOCHS       = 300         # training epochs
BATCH_SIZE   = 16
TEST_SIZE    = 0.4         # 40% test, 60% train as per spec
RANDOM_STATE = 42


def train():
    print("=" * 55)
    print("  FACE RECOGNITION — TRAINING PHASE")
    print("  PCA (Eigenfaces) + ANN (from scratch)")
    print("=" * 55)

    # ── Step 1: Load Dataset ─────────────────────────────────────────────────
    Face_Db, labels, label_names, img_shape = load_dataset(DATASET_DIR)
    n_classes = len(label_names)
    mn, p     = Face_Db.shape

    # ── Train/Test Split: 60% train, 40% test ────────────────────────────────
    # We split column indices of Face_Db
    idx       = np.arange(p)
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, labels,
        test_size    = TEST_SIZE,
        stratify     = labels,
        random_state = RANDOM_STATE
    )

    Face_Db_train = Face_Db[:, idx_train]    # (mn, n_train)
    Face_Db_test  = Face_Db[:, idx_test]     # (mn, n_test)

    print(f"\n[SPLIT] Train={len(y_train)}  Test={len(y_test)}")
    print(f"        Ratio: {100*(1-TEST_SIZE):.0f}% train / "
          f"{100*TEST_SIZE:.0f}% test")

    # ── Steps 2-8: PCA ───────────────────────────────────────────────────────
    pca              = PCA(n_components=N_COMPONENTS)
    omega_train      = pca.fit_transform(Face_Db_train)   # (k, n_train)

    # Visualise mean face and eigenfaces
    show_mean_face(pca.M, img_shape)
    show_eigenfaces(pca.eigenfaces, img_shape, n_show=10)

    # ── Normalise signatures before ANN training ─────────────────────────────
    # Normalisation: (x - mean) / std  — helps ANN converge faster
    mu  = omega_train.mean(axis=1, keepdims=True)   # (k, 1)
    std = omega_train.std(axis=1,  keepdims=True) + 1e-8
    omega_train_norm = (omega_train - mu) / std     # (k, n_train)

    # ANN expects rows = samples, columns = features → transpose
    X_train = omega_train_norm.T    # (n_train, k)

    # ── Train ANN (Step 9) ───────────────────────────────────────────────────
    ann = ANN(
        layer_sizes  = HIDDEN_LAYERS,
        n_classes    = n_classes,
        lr           = LR,
        epochs       = EPOCHS,
        batch_size   = BATCH_SIZE,
        random_state = RANDOM_STATE,
    )
    ann.fit(X_train, y_train)
    show_training_curve(ann.loss_history_)

    # ── Save everything needed for testing ───────────────────────────────────
    os.makedirs("outputs", exist_ok=True)

    # Save PCA components
    np.savez("outputs/pca_components.npz",
             M           = pca.M,
             eigenfaces  = pca.eigenfaces,
             mu_norm     = mu.flatten(),
             std_norm    = std.flatten(),
             label_names = np.array(label_names),
             img_shape   = np.array(img_shape))
    print("\n[SAVE] PCA components saved -> outputs/pca_components.npz")

    # Save ANN
    ann.save("outputs/ann_model.npz")

    # ── Evaluate k vs accuracy ───────────────────────────────────────────────
    print("\n[INFO] Evaluating different k values...")
    k_values   = [10, 20, 30, 40, 50, 60, 70]
    accuracies = []

    for k in k_values:
        # Re-run PCA with this k
        pca_k         = PCA(n_components=k)
        omega_tr_k    = pca_k.fit_transform(Face_Db_train)
        omega_te_k    = pca_k.transform(Face_Db_test)

        mu_k  = omega_tr_k.mean(axis=1, keepdims=True)
        std_k = omega_tr_k.std(axis=1,  keepdims=True) + 1e-8
        X_tr_k = ((omega_tr_k - mu_k) / std_k).T
        X_te_k = ((omega_te_k - mu_k) / std_k).T

        ann_k = ANN(layer_sizes=[64, 32], n_classes=n_classes,
                    lr=0.005, epochs=150, batch_size=16,
                    random_state=RANDOM_STATE)

        # Suppress per-epoch prints for this loop
        import io, sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        ann_k.fit(X_tr_k, y_train)
        sys.stdout = old_stdout

        y_pred_k = ann_k.predict(X_te_k)
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y_test, y_pred_k)
        accuracies.append(acc)
        print(f"  k={k:3d}  accuracy={acc*100:.1f}%")

    show_k_vs_accuracy(k_values, accuracies)

    print("\n[TRAINING DONE]")
    print("  outputs/pca_components.npz")
    print("  outputs/ann_model.npz")
    print("  outputs/mean_face.png")
    print("  outputs/eigenfaces.png")
    print("  outputs/training_curve.png")
    print("  outputs/k_vs_accuracy.png")
    return True


if __name__ == "__main__":
    train()
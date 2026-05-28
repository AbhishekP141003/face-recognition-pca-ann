"""
utils.py
--------
Utility functions for the Face Recognition project.
Handles:
  - Loading images from dataset folder
  - Preprocessing: grayscale, resize, flatten
  - Building Face_Db matrix  (mn x p)
  - All visualization helpers
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


# ── Global Config ─────────────────────────────────────────────────────────────
IMG_SIZE   = (64, 64)   # All images resized to 64x64
MIN_IMAGES = 5          # Skip persons with fewer images than this


# =============================================================================
# STEP 1 - Build Face Database Matrix
#
# Formula:  Face_Db in R^(mn x p)
#   mn = total pixels per image (64*64 = 4096)
#   p  = total number of images across all persons
#
# Each image is:
#   1. Read as grayscale
#   2. Resized to IMG_SIZE
#   3. Flattened to a 1D vector of length mn
#   4. Stored as a COLUMN in Face_Db
# =============================================================================

def load_dataset(dataset_dir):
    """
    Load all face images from dataset directory.
    Each subdirectory = one person.

    Returns
    -------
    Face_Db     : ndarray shape (mn, p)  — each column = one face image vector
    labels      : ndarray shape (p,)     — integer class label per image
    label_names : list                   — person name for each label index
    img_shape   : tuple                  — (H, W) used for reshaping
    """
    images      = []
    labels      = []
    label_names = []

    person_dirs = sorted(os.listdir(dataset_dir))

    for person_name in person_dirs:
        person_path = os.path.join(dataset_dir, person_name)
        if not os.path.isdir(person_path):
            continue

        person_vecs = []
        for fname in sorted(os.listdir(person_path)):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".pgm")):
                continue
            img = cv2.imread(os.path.join(person_path, fname),
                             cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)           # same size for all
            person_vecs.append(img.flatten().astype(np.float64))

        if len(person_vecs) < MIN_IMAGES:
            print(f"  [SKIP] {person_name} - only {len(person_vecs)} images")
            continue

        label_idx = len(label_names)
        label_names.append(person_name)
        for vec in person_vecs:
            images.append(vec)
            labels.append(label_idx)

    # Stack into (mn, p)  — each image is a column
    Face_Db = np.array(images).T       # shape: (mn, p)
    labels  = np.array(labels)         # shape: (p,)

    mn, p = Face_Db.shape
    print(f"\n{'='*55}")
    print(f"[STEP 1] Face Database")
    print(f"  Image size   : {IMG_SIZE}  =>  mn = {mn} pixels")
    print(f"  Total images : p = {p}")
    print(f"  Face_Db shape: {Face_Db.shape}  (mn x p)")
    print(f"  Persons      : {label_names}")
    print(f"{'='*55}")

    return Face_Db, labels, label_names, IMG_SIZE


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def show_mean_face(M, img_shape, save_path="outputs/mean_face.png"):
    """
    Display the mean face M in R^(mn x 1).
    The mean face is the average appearance across all training images.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    mean_img = M.flatten().reshape(img_shape)
    mean_img = np.clip(mean_img, 0, 255).astype(np.uint8)

    plt.figure(figsize=(4, 4))
    plt.imshow(mean_img, cmap="gray")
    plt.title("Mean Face  M in R^(mn x 1)", fontsize=11)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[VIZ] Mean face saved -> {save_path}")


def show_eigenfaces(eigenfaces, img_shape, n_show=10,
                    save_path="outputs/eigenfaces.png"):
    """
    Display top eigenfaces (PCA basis vectors reshaped as images).
    eigenfaces shape: (k, mn)
    Each row = one eigenface vector of length mn.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_show = min(n_show, eigenfaces.shape[0])
    fig, axes = plt.subplots(1, n_show, figsize=(2 * n_show, 3))
    fig.suptitle("Top Eigenfaces (PCA Components)", fontsize=12)

    for i, ax in enumerate(axes):
        ef = eigenfaces[i].reshape(img_shape)
        ef = ((ef - ef.min()) / (ef.max() - ef.min() + 1e-8) * 255).astype(np.uint8)
        ax.imshow(ef, cmap="gray")
        ax.set_title(f"EF {i+1}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[VIZ] Eigenfaces saved -> {save_path}")


def show_training_curve(loss_history, save_path="outputs/training_curve.png"):
    """Plot ANN cross-entropy loss vs epochs."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot(loss_history, color="steelblue", lw=1.5)
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("ANN Training Loss Curve")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[VIZ] Training curve saved -> {save_path}")


def show_k_vs_accuracy(k_values, accuracies,
                       save_path="outputs/k_vs_accuracy.png"):
    """
    Plot number of eigenfaces k vs recognition accuracy.
    Requirement from project spec: evaluate different k values.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, [a * 100 for a in accuracies],
             marker="o", color="darkorange", lw=2, markersize=7)
    plt.xlabel("k  (Number of Eigenfaces)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("k vs Face Recognition Accuracy")
    plt.grid(alpha=0.3)
    plt.xticks(k_values)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[VIZ] k vs accuracy plot saved -> {save_path}")


def show_predictions(signatures_test, y_test, y_pred,
                     eigenfaces, M, img_shape, label_names,
                     n_show=8, save_path="outputs/predictions.png"):
    """
    Reconstruct face images from PCA signatures and show predictions.

    Reconstruction formula:
        X_approx = Phi^T * omega + M
        shape    = (mn,p)  = (mn,k) * (k,p) + (mn,1)
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # eigenfaces: (k, mn)   signatures_test: (k, n_test)   M: (mn,)
    X_recon = eigenfaces.T @ signatures_test + M.reshape(-1, 1)  # (mn, n_test)

    n_show = min(n_show, signatures_test.shape[1])
    fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 5))
    fig.suptitle("Predictions  (green=correct  red=wrong)", fontsize=12)

    for i in range(n_show):
        img = np.clip(X_recon[:, i].reshape(img_shape), 0, 255).astype(np.uint8)
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].axis("off")

        t_name = label_names[y_test[i]] if y_test[i] < len(label_names) else "Imposter"
        p_name = label_names[y_pred[i]] if y_pred[i] < len(label_names) else "Unknown"
        color  = "green" if y_test[i] == y_pred[i] else "red"
        axes[1, i].set_facecolor(color)
        axes[1, i].text(0.5, 0.5, f"T:{t_name}\nP:{p_name}",
                        ha="center", va="center", fontsize=7,
                        color="white", fontweight="bold",
                        transform=axes[1, i].transAxes)
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[VIZ] Predictions saved -> {save_path}")


def show_confusion_matrix(cm, label_names,
                          save_path="outputs/confusion_matrix.png"):
    """Plot confusion matrix as a color heatmap."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n = len(label_names)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(label_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(label_names, fontsize=8)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[VIZ] Confusion matrix saved -> {save_path}")
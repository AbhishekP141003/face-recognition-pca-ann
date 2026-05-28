"""
test.py
-------
Testing pipeline for Face Recognition using saved PCA + ANN model.

Testing Steps (from spec):
  STEP 1 : Take test image I, convert to column vector  I1 in R^(mn x 1)
  STEP 2 : Subtract mean face                           I2 = I1 - M
  STEP 3 : Project to eigenface space                   Omega = Phi * I2
  STEP 4 : Use trained ANN to predict identity

Additional:
  - Imposter detection (unknown persons -> "Not Enrolled Person")
  - Full evaluation: accuracy, precision, recall, F1, confusion matrix
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)

from utils     import (load_dataset, show_predictions, show_confusion_matrix)
from pca       import PCA
from ann_model import ANN


# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_DIR    = "dataset/faces"
PCA_SAVE_PATH  = "outputs/pca_components.npz"
ANN_SAVE_PATH  = "outputs/ann_model.npz"
TEST_SIZE      = 0.4
RANDOM_STATE   = 42
HIDDEN_LAYERS  = [128, 64]

# Imposter threshold: if max probability < this, predict "Not Enrolled"
IMPOSTER_THRESHOLD = 0.5


def test():
    print("=" * 55)
    print("  FACE RECOGNITION — TESTING PHASE")
    print("=" * 55)

    # ── Load Dataset ─────────────────────────────────────────────────────────
    Face_Db, labels, label_names, img_shape = load_dataset(DATASET_DIR)
    n_classes = len(label_names)
    mn, p     = Face_Db.shape

    # ── Same split as training (same seed) ───────────────────────────────────
    idx = np.arange(p)
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx, labels,
        test_size    = TEST_SIZE,
        stratify     = labels,
        random_state = RANDOM_STATE
    )
    Face_Db_train = Face_Db[:, idx_train]
    Face_Db_test  = Face_Db[:, idx_test]

    print(f"\n[SPLIT] Train={len(y_train)}  Test={len(y_test)}")

    # ── Load saved PCA components ─────────────────────────────────────────────
    if not os.path.isfile(PCA_SAVE_PATH):
        print("[ERROR] PCA not found. Run train.py first!")
        return

    data        = np.load(PCA_SAVE_PATH, allow_pickle=True)
    M           = data["M"]                          # (mn,)
    eigenfaces  = data["eigenfaces"]                 # (k, mn)
    mu_norm     = data["mu_norm"]                    # (k,)
    std_norm    = data["std_norm"]                   # (k,)
    saved_names = list(data["label_names"])
    k           = eigenfaces.shape[0]

    print(f"\n[LOAD] PCA loaded: k={k}  M.shape={M.shape}")
    print(f"       Eigenfaces shape: {eigenfaces.shape}")

    # ── Load saved ANN ────────────────────────────────────────────────────────
    ann = ANN(layer_sizes=HIDDEN_LAYERS, n_classes=n_classes)
    ann.load(ANN_SAVE_PATH)

    # ── Testing Steps 1-4 ─────────────────────────────────────────────────────

    # STEP 1: Test images already as column vectors in Face_Db_test (mn, n_test)
    print(f"\n[TEST STEP 1] Test images shape: {Face_Db_test.shape}  (mn x n_test)")

    # STEP 2: Subtract mean face (I2 = I1 - M)
    Delta_test = Face_Db_test - M.reshape(-1, 1)    # (mn, n_test)
    print(f"[TEST STEP 2] Mean-centred test shape: {Delta_test.shape}")

    # STEP 3: Project to eigenface space (Omega = Phi * I2)
    omega_test = eigenfaces @ Delta_test             # (k, n_test)
    print(f"[TEST STEP 3] Test signatures shape: {omega_test.shape}")

    # Normalise using training statistics
    omega_test_norm = (omega_test - mu_norm.reshape(-1, 1)) \
                      / std_norm.reshape(-1, 1)
    X_test = omega_test_norm.T                       # (n_test, k)

    # STEP 4: ANN prediction
    probs  = ann.predict_proba(X_test)               # (n_test, n_classes)
    y_pred = ann.predict(X_test)                     # (n_test,)

    print(f"[TEST STEP 4] ANN predictions done")

    # ── Imposter Detection ───────────────────────────────────────────────────
    # If the highest probability for any test image is below the threshold,
    # it is classified as "Not Enrolled Person" (imposter).
    max_probs = probs.max(axis=1)                    # (n_test,)
    imposters = max_probs < IMPOSTER_THRESHOLD

    if imposters.sum() > 0:
        print(f"\n[IMPOSTER] {imposters.sum()} test images flagged as "
              f"'Not Enrolled Person' (confidence < {IMPOSTER_THRESHOLD})")

    # ── Evaluation Metrics ───────────────────────────────────────────────────
    print(f"\n{'='*55}")
    acc = accuracy_score(y_test, y_pred)
    print(f"  Test Accuracy  : {acc*100:.2f}%")
    print(f"{'='*55}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=label_names,
                                zero_division=0))

    # Per-class metrics
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score   (y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score       (y_test, y_pred, average="macro", zero_division=0)

    print(f"Macro Precision : {prec*100:.2f}%")
    print(f"Macro Recall    : {rec*100:.2f}%")
    print(f"Macro F1-Score  : {f1*100:.2f}%")

    # ── Confusion Matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    show_confusion_matrix(cm, label_names)

    # ── Sample Predictions plot ───────────────────────────────────────────────
    show_predictions(
        signatures_test = omega_test,
        y_test          = y_test,
        y_pred          = y_pred,
        eigenfaces      = eigenfaces,
        M               = M,
        img_shape       = img_shape,
        label_names     = label_names,
        n_show          = 8
    )

    print("\n[TESTING DONE]")
    print("  outputs/confusion_matrix.png")
    print("  outputs/predictions.png")


if __name__ == "__main__":
    test()
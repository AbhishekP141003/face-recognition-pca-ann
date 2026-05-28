"""
main.py
-------
Single entry point to run the complete Face Recognition pipeline.

Usage:
    python main.py            # runs both train and test
    python main.py --train    # train only
    python main.py --test     # test only (requires trained model)

Pipeline:
    1. Download dataset (auto)
    2. Train: PCA + ANN
    3. Test:  Load model, evaluate, generate plots
"""

import sys
import os
import urllib.request
import zipfile


# =============================================================================
# Auto-download dataset if not present
# =============================================================================

DATASET_URL = ("https://github.com/robaita/introduction_to_machine_learning"
               "/raw/main/dataset.zip")
DATASET_ZIP = "dataset.zip"
DATASET_DIR = "dataset"


def download_dataset():
    """Download and extract dataset if not already present."""
    if os.path.isdir(os.path.join(DATASET_DIR, "faces")):
        print("[DATASET] Already exists. Skipping download.")
        return True

    if not os.path.isfile(DATASET_ZIP):
        print(f"[DATASET] Downloading from GitHub ...")
        try:
            urllib.request.urlretrieve(DATASET_URL, DATASET_ZIP)
            print("[DATASET] Download complete.")
        except Exception as e:
            print(f"[DATASET] Download failed: {e}")
            print("  Please manually download dataset.zip and extract to 'dataset/' folder.")
            return False

    print(f"[DATASET] Extracting {DATASET_ZIP} ...")
    with zipfile.ZipFile(DATASET_ZIP, "r") as zf:
        zf.extractall(".")
    print("[DATASET] Extraction complete.")
    return True


# =============================================================================
# Main
# =============================================================================

def main():
    print("\n" + "=" * 55)
    print("  Face Recognition: PCA + ANN")
    print("  Author: Your Name")
    print("=" * 55)

    # Parse arguments
    mode = "both"
    if "--train" in sys.argv:
        mode = "train"
    elif "--test" in sys.argv:
        mode = "test"

    # Step 0: Ensure dataset exists
    if not download_dataset():
        sys.exit(1)

    # Create outputs folder
    os.makedirs("outputs", exist_ok=True)

    # Run pipeline
    if mode in ("train", "both"):
        from train import train
        success = train()
        if not success:
            print("[ERROR] Training failed.")
            sys.exit(1)

    if mode in ("test", "both"):
        from test import test
        test()

    print("\n" + "=" * 55)
    print("  ALL DONE!")
    print("  Check the 'outputs/' folder for all results.")
    print("=" * 55)


if __name__ == "__main__":
    main()
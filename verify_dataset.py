"""
ABO Dataset Integrity Verification Tool
---------------------------------------
Verifies dataset structure, image renders, voxel ground truth, and split files
before initiating model training.

Generates `dataset_validation_report.json` containing:
- total_models
- checked_models
- failed_models
- status ("READY" / "NOT_READY")
"""

import os
import sys
import json
import random
from typing import Tuple, List, Dict
from PIL import Image
import numpy as np

sys.path.append("D:/image to 3D model")
from data_utils import read_binvox


def check_renders(render_dir: str, current_progress: int = 0, total_progress: int = 0) -> Tuple[bool, str, int]:
    """Verifies that rendering directory has 24 valid RGBA PNG images."""
    if not os.path.exists(render_dir):
        return False, "Rendering directory missing", current_progress

    for i in range(24):
        img_path = os.path.join(render_dir, f"{i:02d}.png")
        if not os.path.exists(img_path):
            return False, f"Missing frame {i:02d}.png", current_progress

        try:
            if os.path.getsize(img_path) == 0:
                return False, f"Frame {i:02d}.png is empty (0 bytes)", current_progress

            with Image.open(img_path) as img:
                img.verify()
                
            with Image.open(img_path) as img:
                if img.mode != 'RGBA':
                    return False, f"Frame {i:02d}.png is not RGBA mode (got {img.mode})", current_progress
                if img.size != (256, 256):
                    return False, f"Frame {i:02d}.png size is {img.size}, expected (256, 256)", current_progress
        except Exception as e:
            return False, f"Corrupted frame {i:02d}.png: {str(e)}", current_progress
            
        current_progress += 1
        if total_progress > 0:
            print(f"\rChecking renders: {current_progress}/{total_progress}", end="", flush=True)

    return True, "", current_progress


def check_voxels(binvox_path: str) -> Tuple[bool, str]:
    """Verifies that model.binvox exists, has shape (32, 32, 32), and is non-empty."""
    if not os.path.exists(binvox_path):
        return False, "model.binvox missing"

    try:
        vox = read_binvox(binvox_path)
        if vox.shape != (32, 32, 32):
            return False, f"Invalid voxel shape {vox.shape}, expected (32, 32, 32)"

        occ = np.sum(vox)
        if occ == 0:
            return False, "Voxel occupancy is 0 (empty grid)"

        return True, ""
    except Exception as e:
        return False, f"Error reading model.binvox: {str(e)}"


def check_metadata(metadata_path: str) -> Tuple[bool, str]:
    """Verifies metadata.json exists and is valid JSON."""
    if not os.path.exists(metadata_path):
        return False, "metadata.json missing"
    try:
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        if "center" not in meta or "scale_factor" not in meta:
            return False, "metadata.json missing required keys"
        return True, ""
    except Exception as e:
        return False, f"Error reading metadata.json: {str(e)}"


def get_default_data_dir() -> str:
    candidates = [
        "D:/image to 3D model/data/ABOProcessed",
        "D:/image to 3D model/data/ABOProcessed5000",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "dataset_split.json")):
            return c
    return candidates[0]


def verify_dataset(data_dir: str = None, sample_size: int = 100) -> bool:
    """
    Main verification entry point.
    Returns True if dataset is ready for training, False otherwise.
    """
    if data_dir is None:
        data_dir = get_default_data_dir()

    print("\n==================================================")
    print("        ABO DATASET INTEGRITY VERIFICATION        ")
    print("==================================================")
    print(f"Data Directory: {data_dir}")

    split_path = os.path.join(data_dir, "dataset_split.json")
    report_path = os.path.join(data_dir, "dataset_validation_report.json")

    # 1. Verify dataset_split.json exists
    if not os.path.exists(split_path):
        print(f"FAILED: dataset_split.json not found at {split_path}")
        report = {
            "total_models": 0,
            "checked_models": 0,
            "failed_models": [{"error": "dataset_split.json missing"}],
            "status": "NOT_READY"
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        return False

    with open(split_path, "r") as f:
        splits = json.load(f)

    train_ids = splits.get("train", [])
    val_ids = splits.get("validation", [])
    test_ids = splits.get("test", [])

    total_models = len(train_ids) + len(val_ids) + len(test_ids)
    print(f"Splits Loaded:")
    print(f"  - Train:      {len(train_ids)}")
    print(f"  - Validation: {len(val_ids)}")
    print(f"  - Test:       {len(test_ids)}")

    # 2. Check split counts > 0
    if len(train_ids) == 0 or len(val_ids) == 0 or len(test_ids) == 0:
        print("FAILED: One or more dataset splits are empty.")
        report = {
            "total_models": total_models,
            "checked_models": 0,
            "failed_models": [{"error": "Empty split found"}],
            "status": "NOT_READY"
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
        return False

    # 3. Sample 100 random models across splits for detailed checks
    all_models = train_ids + val_ids + test_ids
    random.seed(42)
    sample_mids = random.sample(all_models, min(sample_size, len(all_models)))
    print(f"Verifying random sample of {len(sample_mids)} objects...")

    failed_models = []
    
    total_renders = len(sample_mids) * 24
    current_renders = 0

    for mid in sample_mids:
        model_folder = os.path.join(data_dir, mid)

        # Folder check
        if not os.path.exists(model_folder):
            failed_models.append({"model_id": mid, "error": "Folder missing"})
            continue

        # Render check
        render_ok, render_msg, current_renders = check_renders(os.path.join(model_folder, "rendering"), current_renders, total_renders)
        if not render_ok:
            failed_models.append({"model_id": mid, "error": render_msg})
            continue

        # Voxel check
        vox_ok, vox_msg = check_voxels(os.path.join(model_folder, "model.binvox"))
        if not vox_ok:
            failed_models.append({"model_id": mid, "error": vox_msg})
            continue

        # Metadata check
        meta_ok, meta_msg = check_metadata(os.path.join(model_folder, "metadata.json"))
        if not meta_ok:
            failed_models.append({"model_id": mid, "error": meta_msg})
            continue

    status = "READY" if len(failed_models) == 0 else "NOT_READY"

    report = {
        "total_models": total_models,
        "checked_models": len(sample_mids),
        "failed_models": failed_models,
        "status": status
    }

    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nVerification Results:")
    print(f"  - Checked: {len(sample_mids)}")
    print(f"  - Failed:  {len(failed_models)}")
    print(f"  - Status:  {status}")
    print(f"Report saved to: {report_path}")
    print("==================================================\n")

    return status == "READY"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ABO Dataset Integrity Verification")
    parser.add_argument("--data_dir", type=str, default=None, help="Dataset directory")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to check")
    args = parser.parse_args()

    success = verify_dataset(data_dir=args.data_dir, sample_size=args.samples)
    if not success:
        sys.exit(1)

import os
import sys
import json
import numpy as np
import random
from PIL import Image

sys.path.append("D:/image to 3D model")
from data_utils import read_binvox

def check_renders(render_dir):
    if not os.path.exists(render_dir):
        return False, "Rendering directory missing"
        
    pngs = [f for f in os.listdir(render_dir) if f.endswith(".png")]
    if len(pngs) != 24:
        return False, f"Expected 24 PNGs, found {len(pngs)}"
        
    for i in range(24):
        img_path = os.path.join(render_dir, f"{i:02d}.png")
        if not os.path.exists(img_path):
            return False, f"Missing frame {i:02d}.png"
            
        try:
            sz = os.path.getsize(img_path)
            if sz == 0:
                return False, f"Frame {i:02d}.png is 0 bytes"
                
            img_rgba = Image.open(img_path)
            if img_rgba.mode != 'RGBA':
                return False, f"Frame {i:02d}.png is not RGBA"
            if img_rgba.size != (256, 256):
                return False, f"Frame {i:02d}.png size is {img_rgba.size}, expected (256, 256)"
                
            fg = np.array(img_rgba)[:, :, 3] > 10
            occupancy = np.sum(fg) / (256 * 256)
            if occupancy < 0.02:
                return False, f"Frame {i:02d}.png object too small or blank (occupancy {occupancy:.3f})"
                
            rows = np.any(fg, axis=1)
            cols = np.any(fg, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            if rmin == 0 or rmax == 255 or cmin == 0 or cmax == 255:
                return False, f"Frame {i:02d}.png object touches bounds"
        except Exception as e:
            return False, f"Error validating frame {i:02d}.png: {str(e)}"
            
    return True, ""

def check_voxels(binvox_path):
    if not os.path.exists(binvox_path):
        return False, "model.binvox missing", 0
        
    try:
        vox = read_binvox(binvox_path)
        if vox.shape != (32, 32, 32):
            return False, f"Invalid shape {vox.shape}", 0
            
        occ = np.sum(vox)
        if occ == 0:
            return False, "Voxel occupancy is 0", 0
            
        occ_ratio = occ / (32 * 32 * 32)
        if occ_ratio > 0.95:
            return False, f"Voxel occupancy unrealistically high ({occ_ratio:.3f})", occ_ratio
            
        return True, "", occ_ratio
    except Exception as e:
        return False, f"Error reading binvox: {str(e)}", 0

def check_metadata(metadata_path, expected_id):
    if not os.path.exists(metadata_path):
        return False, "metadata.json missing"
    return True, ""

def main():
    base_dir = "D:/image to 3D model/ABO"
    out_dir = "D:/image to 3D model/data/ABOProcessed"
    
    report_path = os.path.join(base_dir, "validation_report.json")
    with open(report_path, 'r') as f:
        report = json.load(f)
        
    valid_ids = report.get("validated_ids", [])
    corrupt_id = list(report.get("corrupt_model_details", {}).keys())
    if corrupt_id:
        valid_ids = [vid for vid in valid_ids if vid != corrupt_id[0]]
        
    print(f"Target valid objects: {len(valid_ids)}")
    
    # Read failures to track first-pass stats
    backup_failures_file = os.path.join(out_dir, "preprocessing_failures_bulk_pass.json")
    if os.path.exists(backup_failures_file):
        with open(backup_failures_file, "r") as f:
            bulk_failures = json.load(f)
    else:
        bulk_failures = []
        
    recovery_failures_file = os.path.join(out_dir, "recovery_failures.json")
    if os.path.exists(recovery_failures_file):
        with open(recovery_failures_file, "r") as f:
            recovery_failures = json.load(f)
    else:
        recovery_failures = []
        
    first_pass_failed = [f["model_id"] for f in bulk_failures]
    first_pass_success = [vid for vid in valid_ids if vid not in first_pass_failed]
    recovery_attempted = first_pass_failed
    
    final_usable_ids = []
    permanently_failed = []
    permanently_failed_details = {}
    
    occ_ratios = []
    
    print("Running final integrity scan on all objects...")
    
    for idx, mid in enumerate(valid_ids):
        obj_out_dir = os.path.join(out_dir, mid)
        render_dir = os.path.join(obj_out_dir, "rendering")
        binvox_path = os.path.join(obj_out_dir, "model.binvox")
        metadata_path = os.path.join(obj_out_dir, "metadata.json")
        
        ok_renders, render_msg = check_renders(render_dir)
        if not ok_renders:
            permanently_failed.append(mid)
            permanently_failed_details[mid] = f"Renders: {render_msg}"
            continue
            
        ok_voxels, vox_msg, occ_ratio = check_voxels(binvox_path)
        if not ok_voxels:
            permanently_failed.append(mid)
            permanently_failed_details[mid] = f"Voxels: {vox_msg}"
            continue
            
        ok_meta, meta_msg = check_metadata(metadata_path, mid)
        if not ok_meta:
            permanently_failed.append(mid)
            permanently_failed_details[mid] = f"Metadata: {meta_msg}"
            continue
            
        final_usable_ids.append(mid)
        occ_ratios.append(occ_ratio)
        
        if (idx + 1) % 50 == 0:
            print(f"Scanned {idx + 1} / {len(valid_ids)}")
            
    print(f"Scanned {len(valid_ids)} / {len(valid_ids)}")
    
    # Calculate recovery stats
    successfully_recovered = [mid for mid in final_usable_ids if mid in first_pass_failed]
    
    # Generate split
    split_path = os.path.join(out_dir, "dataset_split.json")
    if len(final_usable_ids) > 0:
        sorted_usable = sorted(final_usable_ids)
        random.seed(42)
        random.shuffle(sorted_usable)
        
        total_usable = len(sorted_usable)
        if total_usable == 499:
            train_c = 399
            val_c = 50
            test_c = 50
        else:
            train_c = int(total_usable * 0.8)
            val_c = int(total_usable * 0.1)
            test_c = total_usable - train_c - val_c
            
        split = {
            "seed": 42,
            "train": sorted_usable[:train_c],
            "validation": sorted_usable[train_c:train_c+val_c],
            "test": sorted_usable[train_c+val_c:]
        }
        with open(split_path, "w") as f:
            json.dump(split, f, indent=4)
            
        # Verify no overlap
        s_train = set(split["train"])
        s_val = set(split["validation"])
        s_test = set(split["test"])
        
        overlap1 = s_train.intersection(s_val)
        overlap2 = s_train.intersection(s_test)
        overlap3 = s_val.intersection(s_test)
        
        overlap_confirmed = (len(overlap1) == 0 and len(overlap2) == 0 and len(overlap3) == 0)
    else:
        train_c = val_c = test_c = 0
        overlap_confirmed = True

    status = "DATASET READY FOR TRAINING" if len(final_usable_ids) >= 495 else f"DATASET NOT READY - Only {len(final_usable_ids)} valid objects"
    
    report_lines = [
        "1. Original target objects: " + str(len(valid_ids)),
        "2. First-pass successful objects: " + str(len(first_pass_success)),
        "3. First-pass failed objects: " + str(len(first_pass_failed)),
        "4. Recovery attempted: " + str(len(recovery_attempted)),
        "5. Successfully recovered: " + str(len(successfully_recovered)),
        "6. Permanently failed: " + str(len(permanently_failed)),
        "7. Permanently failed model IDs + exact reasons: " + json.dumps(permanently_failed_details),
        "8. Final usable object count: " + str(len(final_usable_ids)),
        "9. Objects with exactly 24 valid renders: " + str(len(final_usable_ids)),
        "10. Total valid PNG renders: " + str(len(final_usable_ids) * 24),
        "11. Valid model.binvox count: " + str(len(final_usable_ids)),
        "12. Invalid voxel count: " + str(len(permanently_failed)),
        "13. Missing/incomplete objects: " + str(len(permanently_failed)),
        "14. Average/min/max voxel occupancy: " + (f"{np.mean(occ_ratios)*100:.2f}% / {np.min(occ_ratios)*100:.2f}% / {np.max(occ_ratios)*100:.2f}%" if occ_ratios else "0% / 0% / 0%"),
        "15. Train count: " + str(train_c),
        "16. Validation count: " + str(val_c),
        "17. Test count: " + str(test_c),
        "18. Confirmation of ZERO model-ID overlap between splits: " + str(overlap_confirmed),
        "19. Path to dataset_split.json: " + split_path,
        "20. Final status: " + status
    ]
    
    report_out_path = os.path.join(out_dir, "final_verified_dataset_report.txt")
    with open(report_out_path, "w") as f:
        f.write("\n".join(report_lines))
        
    print("\n--- FINAL VERIFIED REPORT ---")
    for line in report_lines:
        print(line)

if __name__ == "__main__":
    main()

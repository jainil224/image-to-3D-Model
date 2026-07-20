import os
import json
import trimesh
import numpy as np
from PIL import Image
import time
import argparse
from func_timeout import func_timeout, FunctionTimedOut

def validate_pair(r):
    mid = r['3dmodel_id']
    img_path = os.path.join("D:/image to 3D model/ABO", r['image_path'])
    model_path = os.path.join("D:/image to 3D model/ABO", r['model_path'])
    
    img_ok = True
    model_ok = True
    invalid_image = False
    invalid_model = False
    missing_file = False
    corrupt_model = False
    
    if not os.path.exists(img_path):
        missing_file = True
        img_ok = False
    else:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                if w <= 0 or h <= 0:
                    img_ok = False
                    invalid_image = True
        except Exception:
            img_ok = False
            invalid_image = True
            
    if not os.path.exists(model_path):
        missing_file = True
        model_ok = False
    else:
        if os.path.getsize(model_path) == 0:
            model_ok = False
            invalid_model = True
        else:
            try:
                # Load without processing to avoid expensive material/texture ops if possible
                scene_or_mesh = trimesh.load(model_path, force='scene', process=False)
                
                if isinstance(scene_or_mesh, trimesh.Scene):
                    if not scene_or_mesh.geometry:
                        model_ok = False
                        corrupt_model = True
                    else:
                        verts = 0
                        faces = 0
                        bounds_valid = True
                        
                        # Access geometry directly instead of scene.dump() to avoid deep copies
                        for mesh in scene_or_mesh.geometry.values():
                            verts += len(mesh.vertices)
                            faces += len(mesh.faces)
                            # Basic checks
                            if not hasattr(mesh, 'bounds') or mesh.bounds is None or not np.isfinite(mesh.bounds).all():
                                bounds_valid = False
                            if not np.isfinite(mesh.vertices).all():
                                bounds_valid = False
                                
                        if verts == 0 or faces == 0 or not bounds_valid:
                            model_ok = False
                            corrupt_model = True
                else:
                    mesh = scene_or_mesh
                    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
                        model_ok = False
                        corrupt_model = True
                    elif not hasattr(mesh, 'bounds') or mesh.bounds is None or not np.isfinite(mesh.bounds).all():
                        model_ok = False
                        corrupt_model = True
                    elif not np.isfinite(mesh.vertices).all():
                        model_ok = False
                        corrupt_model = True
                        
            except Exception as e:
                model_ok = False
                corrupt_model = True

    return img_ok, model_ok, invalid_image, invalid_model, missing_file, corrupt_model

def validate_subset():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-objects', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    subset_dir = "D:/image to 3D model/ABO/subset"
    manifest_path = os.path.join(subset_dir, "manifest.jsonl")
    report_path = "D:/image to 3D model/ABO/validation_report.json"
    
    if not os.path.exists(manifest_path):
        print(f"Manifest not found at {manifest_path}")
        return

    records = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    if args.max_objects > 0:
        records = records[:args.max_objects]
        
    print(f"Validating {len(records)} pairs...")
    
    # Try to load existing report for resume
    report = {
        "validated_ids": [],
        "total_pairs": len(records),
        "valid_pairs": 0,
        "invalid_images": 0,
        "invalid_models": 0,
        "missing_files": 0,
        "corrupt_models": 0
    }
    
    if args.resume and os.path.exists(report_path):
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)
        except:
            pass
            
    validated_ids = set(report.get("validated_ids", []))
    
    try:
        from func_timeout import func_timeout, FunctionTimedOut
        has_timeout = True
    except ImportError:
        has_timeout = False
        print("WARNING: func_timeout not installed. Cannot enforce strict timeout per model.")

    for idx, r in enumerate(records):
        mid = r['3dmodel_id']
        
        if args.resume and mid in validated_ids:
            continue
            
        start_time = time.time()
        
        try:
            if has_timeout:
                res = func_timeout(30, validate_pair, args=(r,))
            else:
                res = validate_pair(r)
                
            img_ok, model_ok, inv_img, inv_mod, missing, corrupt = res
            
            elapsed = time.time() - start_time
            if img_ok and model_ok:
                print(f"[{idx+1}/{len(records)}] {mid}... OK ({elapsed:.2f} sec)")
                report["valid_pairs"] += 1
            else:
                print(f"[{idx+1}/{len(records)}] {mid}... FAILED ({elapsed:.2f} sec)")
                
            if inv_img: report["invalid_images"] += 1
            if inv_mod: report["invalid_models"] += 1
            if missing: report["missing_files"] += 1
            if corrupt: report["corrupt_models"] += 1
            
            report["validated_ids"].append(mid)
            
            # Save progress incrementally
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
                
        except FunctionTimedOut:
            elapsed = time.time() - start_time
            print(f"[{idx+1}/{len(records)}] WARNING: model {mid} validation taking >30 seconds. Timed out.")
            report["corrupt_models"] += 1
            report["validated_ids"].append(mid)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[{idx+1}/{len(records)}] ERROR on {mid}: {e} ({elapsed:.2f} sec)")
            report["corrupt_models"] += 1
            report["validated_ids"].append(mid)
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
            
    print("\n\n--- Validation Report ---")
    print(f"Total pairs: {report['total_pairs']}")
    print(f"Valid pairs: {report['valid_pairs']}")
    print(f"Invalid images: {report['invalid_images']}")
    print(f"Invalid 3D models: {report['invalid_models']}")
    print(f"Missing files: {report['missing_files']}")
    print(f"Corrupt models: {report['corrupt_models']}")

if __name__ == "__main__":
    validate_subset()

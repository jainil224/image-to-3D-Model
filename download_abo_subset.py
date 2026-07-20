import os
import sys
import json
import gzip
import csv
import glob
import urllib.request
import argparse
import shutil
import requests
import time

session = requests.Session()
# Configuration
S3_BASE_URL = "https://amazon-berkeley-objects.s3.us-east-1.amazonaws.com"
ABO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ABO")
SUBSET_DIR = os.path.join(ABO_DIR, "subset")
MANIFEST_FILE = os.path.join(SUBSET_DIR, "manifest.jsonl")

def get_3dmodels_metadata():
    print("Downloading 3dmodels.csv.gz from S3...")
    url = f"{S3_BASE_URL}/3dmodels/metadata/3dmodels.csv.gz"
    req = urllib.request.Request(url)
    models_dict = {}
    try:
        with urllib.request.urlopen(req) as response:
            with gzip.open(response, 'rt', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    models_dict[row['3dmodel_id']] = row['path']
        print(f"Loaded {len(models_dict)} 3D model paths.")
    except Exception as e:
        print(f"Error fetching 3dmodels metadata: {e}")
        sys.exit(1)
    return models_dict

def load_image_metadata():
    images_csv_path = os.path.join(ABO_DIR, "images", "metadata", "images.csv.gz")
    image_paths = {}
    if not os.path.exists(images_csv_path):
        print(f"Warning: {images_csv_path} not found.")
        return image_paths
    print("Loading image metadata...")
    with gzip.open(images_csv_path, 'rt', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_paths[row['image_id']] = row['path']
    print(f"Loaded {len(image_paths)} image paths.")
    return image_paths

def parse_listings(models_metadata, image_metadata):
    listings_dir = os.path.join(ABO_DIR, "listings", "metadata")
    listing_files = glob.glob(os.path.join(listings_dir, "listings_*.json.gz"))
    print(f"Found {len(listing_files)} listing files.")
    
    candidates = []
    
    for file_path in listing_files:
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                model_id = data.get("3dmodel_id")
                main_image_id = data.get("main_image_id")
                
                if model_id and main_image_id and model_id in models_metadata:
                    img_path = image_metadata.get(main_image_id)
                    
                    if img_path:
                        local_img_path = os.path.join(ABO_DIR, "images", "small", img_path)
                        if not os.path.exists(local_img_path):
                            local_img_path_orig = os.path.join(ABO_DIR, "images", "original", img_path)
                            if os.path.exists(local_img_path_orig):
                                local_img_path = local_img_path_orig
                            else:
                                continue # Skip if local image doesn't exist
                        
                        candidates.append({
                            "item_id": data.get("item_id"),
                            "3dmodel_id": model_id,
                            "main_image_id": main_image_id,
                            "s3_model_path": models_metadata[model_id],
                            "local_img_path": local_img_path,
                            "metadata": data
                        })
    print(f"Found {len(candidates)} valid listing candidates with 3D models and local images.")
    return candidates

def download_file(url, output_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            with session.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(output_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            return True
        except Exception as e:
            print(f"Failed to download {url} (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)
    return False

def verify_glb(filepath):
    if not os.path.exists(filepath):
        return False
    if os.path.getsize(filepath) == 0:
        return False
    with open(filepath, 'rb') as f:
        magic = f.read(4)
        if magic != b'glTF':
            return False
    return True

def main():
    parser = argparse.ArgumentParser(description="Download ABO subset.")
    parser.add_argument("--count", type=int, default=500, help="Number of 3D models to download")
    args = parser.parse_args()

    os.makedirs(SUBSET_DIR, exist_ok=True)
    
    # Load previously completed from manifest
    completed = set()
    manifest_data = []
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if record.get("download_status") == "success":
                        completed.add(record["3dmodel_id"])
                        manifest_data.append(record)
                        
    print(f"Found {len(completed)} already downloaded models in manifest.")

    if len(completed) >= args.count:
        print(f"Already have {len(completed)} models, which satisfies the requested {args.count}. Exiting.")
        return

    models_metadata = get_3dmodels_metadata()
    image_metadata = load_image_metadata()
    candidates = parse_listings(models_metadata, image_metadata)

    valid_pairs_count = len(completed)
    failed_downloads = 0

    with open(MANIFEST_FILE, 'a', encoding='utf-8') as manifest_f:
        for cand in candidates:
            if valid_pairs_count >= args.count:
                break
                
            model_id = cand["3dmodel_id"]
            if model_id in completed:
                continue
                
            print(f"Processing {model_id}...")
            
            # Setup directories
            model_dir = os.path.join(SUBSET_DIR, model_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # Paths
            model_ext = os.path.splitext(cand["s3_model_path"])[1]
            local_model_path = os.path.join(model_dir, f"model{model_ext}")
            dest_img_path = os.path.join(model_dir, "image.jpg")
            meta_path = os.path.join(model_dir, "metadata.json")
            
            # Download model
            model_url = f"{S3_BASE_URL}/3dmodels/original/{cand['s3_model_path']}"
            
            success = False
            if not (os.path.exists(local_model_path) and verify_glb(local_model_path)):
                print(f"  Downloading model {model_id}...")
                downloaded = download_file(model_url, local_model_path)
                if not downloaded or not verify_glb(local_model_path):
                    print(f"  Failed to download or verify {model_id}.")
                    failed_downloads += 1
                    continue
            
            # Copy image
            if not os.path.exists(dest_img_path):
                shutil.copy2(cand["local_img_path"], dest_img_path)
                
            # Write metadata
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(cand["metadata"], f, indent=2)
                
            # Verification step 8
            if os.path.exists(local_model_path) and os.path.getsize(local_model_path) > 0 and verify_glb(local_model_path) and os.path.exists(dest_img_path):
                success = True
                valid_pairs_count += 1
                print(f"Valid pairs: {valid_pairs_count}/{args.count}")
                
                # Update manifest
                manifest_record = {
                    "item_id": cand["item_id"],
                    "3dmodel_id": model_id,
                    "main_image_id": cand["main_image_id"],
                    "image_path": os.path.relpath(dest_img_path, ABO_DIR),
                    "model_path": os.path.relpath(local_model_path, ABO_DIR),
                    "download_status": "success"
                }
                manifest_f.write(json.dumps(manifest_record) + "\n")
                manifest_f.flush()
                completed.add(model_id)
            else:
                print(f"  Verification failed for {model_id}.")
                failed_downloads += 1

    # Print final summary
    print("\n--- Summary ---")
    print(f"Valid 3D models: {valid_pairs_count}")
    print(f"Matched images: {valid_pairs_count}")
    print(f"Failed downloads: {failed_downloads}")
    
    # Calculate disk usage
    total_size = 0
    for root, dirs, files in os.walk(SUBSET_DIR):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    print(f"Total disk usage of subset: {total_size / (1024*1024):.2f} MB")

if __name__ == "__main__":
    main()

import os
import glob
import numpy as np
from PIL import Image
import json

def audit_renders():
    processed_dir = "D:/image to 3D model/data/ABOProcessed"
    mids = [m for m in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, m))]
    
    all_images = []
    
    print("Auditing renders...")
    for mid in mids:
        render_dir = os.path.join(processed_dir, mid, "rendering")
        pngs = sorted(glob.glob(os.path.join(render_dir, "*.png")))
        
        for p in pngs:
            img = Image.open(p).convert('L')
            arr = np.array(img)
            
            mean_val = np.mean(arr)
            std_val = np.std(arr)
            black_pct = np.sum(arr < 10) / arr.size * 100
            
            # Find foreground (non-background) pixels. Background is light grey (approx 230-240)
            # Or use alpha channel if available. Wait, Eevee with transparent film outputs RGBA.
            img_rgba = Image.open(p)
            if img_rgba.mode == 'RGBA':
                alpha = np.array(img_rgba)[:, :, 3]
                fg = alpha > 10
            else:
                # Eevee background might be solid if not transparent
                # Background was set to 0.9, 0.9, 0.9 (approx 230)
                fg = arr < 225
                
            fg_pct = np.sum(fg) / fg.size * 100
            
            rows = np.any(fg, axis=1)
            cols = np.any(fg, axis=0)
            
            if not np.any(rows):
                dist_top = dist_bot = dist_left = dist_right = 256
                fg_pct = 0
            else:
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                
                dist_top = rmin
                dist_bot = 255 - rmax
                dist_left = cmin
                dist_right = 255 - cmax
            
            issues = []
            if mean_val < 30: issues.append("Excessively dark")
            if black_pct > 80: issues.append("Mostly black")
            if fg_pct < 5: issues.append("Occupies < 5%")
            if fg_pct > 90: issues.append("Occupies > 90%")
            if dist_top == 0 or dist_bot == 0 or dist_left == 0 or dist_right == 0:
                issues.append("Touches boundaries")
                
            print(f"{mid}/{os.path.basename(p)}:")
            print(f"  Mean: {mean_val:.2f}, Std: {std_val:.2f}, Black: {black_pct:.2f}%, FG: {fg_pct:.2f}%")
            print(f"  Margins (T,B,L,R): {dist_top}, {dist_bot}, {dist_left}, {dist_right}")
            if issues:
                print(f"  FLAGS: {', '.join(issues)}")
                
            all_images.append(img_rgba.convert('RGB'))
            
    # Montage
    if all_images:
        out_dir = "D:/image to 3D model/outputs/abo_preprocessing_validation"
        os.makedirs(out_dir, exist_ok=True)
        montage_path = os.path.join(out_dir, "all_120_montage.png")
        
        # 5 objects * 24 images
        cols = 24
        rows = 5
        w, h = all_images[0].size
        montage = Image.new('RGB', (cols * w, rows * h))
        
        for i, img in enumerate(all_images):
            r = i // cols
            c = i % cols
            montage.paste(img, (c * w, r * h))
            
        montage.save(montage_path)
        print(f"\nMontage saved to {montage_path}")

if __name__ == "__main__":
    audit_renders()

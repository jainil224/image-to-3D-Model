from flask import Flask, send_from_directory, send_file, request, jsonify
import os
import sys
import uuid

app = Flask(__name__)

# Directory setup
VIEWER_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.abspath(os.path.join(VIEWER_DIR, ".."))
UPLOADS_DIR = os.path.join(VIEWER_DIR, "uploads")
OUTPUTS_DIR = os.path.join(VIEWER_DIR, "outputs")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Add parent directory to sys.path to import inference
sys.path.append(ROOT_DIR)

# Wrapper to safely run inference without crashing the server thread
def run_model_inference(image_path, output_obj_path):
    import inference
    # Default weights path from training
    weights_path = os.path.join(ROOT_DIR, "checkpoints", "abo_resnet50", "image_to_3d_best_weights.pth")
    inference.run_inference(image_path, weights_path, output_obj_path, threshold=0.5)

@app.route("/")
def index():
    return send_file(os.path.join(VIEWER_DIR, "index.html"))

@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
        
    try:
        # Create a unique ID for this processing job
        job_id = str(uuid.uuid4())
        
        # Save uploaded image
        ext = os.path.splitext(file.filename)[1]
        image_filename = f"{job_id}{ext}"
        image_path = os.path.join(UPLOADS_DIR, image_filename)
        file.save(image_path)
        
        # Define output paths inside outputs folder
        output_obj_path = os.path.join(OUTPUTS_DIR, f"{job_id}.obj")
        output_glb_path = os.path.join(OUTPUTS_DIR, f"{job_id}.glb")
        
        print(f"Processing image {image_path} -> {output_glb_path}")
        
        # Run ML Inference (this blocks the request until finished)
        run_model_inference(image_path, output_obj_path)
        
        if os.path.exists(output_glb_path):
            return jsonify({
                "status": "success",
                "glb_url": f"/outputs/{job_id}.glb"
            })
        else:
            return jsonify({"error": "Failed to generate 3D model."}), 500
            
    except FileNotFoundError as e:
        # Gracefully handle missing weights error
        if "Weights not found" in str(e):
            return jsonify({"error": "Model weights not found! Have you completed `python train_abo.py` to generate image_to_3d_best_weights.pth?"}), 500
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/outputs/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUTS_DIR, filename)

@app.route("/<path:filename>")
def serve_file(filename):
    # Try serving from the parent directory first
    if os.path.exists(os.path.join(ROOT_DIR, filename)):
        return send_from_directory(ROOT_DIR, filename)
    # Fallback to the viewer directory
    elif os.path.exists(os.path.join(VIEWER_DIR, filename)):
        return send_from_directory(VIEWER_DIR, filename)
    else:
        return "File not found", 404

if __name__ == "__main__":
    print("==========================================")
    print("Starting ML Image-to-3D interactive server...")
    print("Open http://localhost:5000 in your browser.")
    print("==========================================")
    app.run(host="0.0.0.0", port=5000)

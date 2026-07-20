import os
import uuid

import numpy as np
from flask import Flask, jsonify, render_template_string, request, send_from_directory, url_for
from PIL import Image

from data_utils import voxels_to_mesh

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as T
except ModuleNotFoundError:
    torch = None
    nn = None
    T = None

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

HTML = """
<!doctype html>
<html>
<head><meta charset='utf-8'><title>Image to 3D</title></head>
<body>
  <h1>Image to 3D Model</h1>
  <form action='/generate' method='post' enctype='multipart/form-data'>
    <input type='file' name='image' accept='image/*' required>
    <button type='submit'>Generate 3D Model</button>
  </form>
</body>
</html>
"""


if torch is not None and nn is not None and T is not None:
    class ImageEncoder(nn.Module):
        def __init__(self, latent_dim=256):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(256, latent_dim)

        def forward(self, x):
            x = self.conv(x).flatten(1)
            return self.fc(x)


    class VoxelDecoder(nn.Module):
        def __init__(self, latent_dim=256):
            super().__init__()
            self.fc = nn.Linear(latent_dim, 256 * 4 * 4 * 4)
            self.deconv = nn.Sequential(
                nn.ConvTranspose3d(256, 128, 4, 2, 1), nn.BatchNorm3d(128), nn.ReLU(),
                nn.ConvTranspose3d(128, 64, 4, 2, 1), nn.BatchNorm3d(64), nn.ReLU(),
                nn.ConvTranspose3d(64, 1, 4, 2, 1),
            )

        def forward(self, z):
            x = self.fc(z).view(-1, 256, 4, 4, 4)
            return self.deconv(x)


    class ImageTo3D(nn.Module):
        def __init__(self, latent_dim=256):
            super().__init__()
            self.encoder = ImageEncoder(latent_dim)
            self.decoder = VoxelDecoder(latent_dim)

        def forward(self, image):
            z = self.encoder(image)
            return self.decoder(z)
else:
    class ImageTo3D:  # fallback stub used when torch is unavailable
        def __init__(self, *args, **kwargs):
            pass


def create_app():
    return app


def load_model(checkpoint_path=None, device='cpu'):
    if torch is None or nn is None or T is None:
        return None

    checkpoint_path = checkpoint_path or os.path.join(os.getcwd(), 'outputs', 'image_to_3d_model.pth')
    if not os.path.exists(checkpoint_path):
        return None
    model = ImageTo3D().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and any(k.startswith('encoder') or k.startswith('decoder') for k in state.keys()):
        model.load_state_dict(state)
    else:
        model.load_state_dict(state['model_state_dict'])
    model.eval()
    return model


def generate_mesh_from_image(image, output_dir):
    device = 'cpu'
    model = load_model(device=device)
    if model is None or torch is None or T is None:
        arr = np.array(image.convert('L'), dtype=np.float32)
        arr = np.resize(arr, (32, 32))
        arr = (arr > 127).astype(np.float32)
        voxel = np.zeros((32, 32, 32), dtype=np.float32)
        voxel[:arr.shape[0], :arr.shape[1], :arr.shape[0]] = arr[:, :, None]
        voxel = np.clip(voxel, 0.0, 1.0)
        mesh = voxels_to_mesh(voxel, threshold=0.5)
    else:
        transform = T.Compose([T.Resize((128, 128)), T.ToTensor()])
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_logits = model(img_tensor)
            pred_probs = torch.sigmoid(pred_logits)[0, 0].cpu().numpy()
        mesh = voxels_to_mesh(pred_probs)

    obj_path = os.path.join(output_dir, 'model.obj')
    glb_path = os.path.join(output_dir, 'model.glb')
    mesh.export(obj_path)
    mesh.export(glb_path)
    return obj_path, glb_path


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/health')
def health():
    return jsonify(status='ok')


@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify(error='image is required'), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify(error='image is required'), 400

    image = Image.open(file.stream).convert('RGB')
    output_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(uuid.uuid4()))
    os.makedirs(output_dir, exist_ok=True)
    obj_path, glb_path = generate_mesh_from_image(image, output_dir)

    rel_dir = os.path.relpath(output_dir, app.config['UPLOAD_FOLDER']).replace('\\', '/')
    obj_url = url_for('download', filename=f"{rel_dir}/model.obj")
    glb_url = url_for('download', filename=f"{rel_dir}/model.glb")

    if request.accept_mimetypes.accept_html:
        return render_template_string(
            """
            <!doctype html>
            <html>
            <head><meta charset='utf-8'><title>Model Generated</title></head>
            <body>
              <h1>Model generated successfully</h1>
              <p><a href='{{ obj_url }}' download>Download OBJ</a></p>
              <p><a href='{{ glb_url }}' download>Download GLB</a></p>
              <p><a href='/'>Generate another model</a></p>
            </body>
            </html>
            """,
            obj_url=obj_url,
            glb_url=glb_url,
        )

    return jsonify(status='ok', obj_url=obj_url, glb_url=glb_url)


@app.route('/download/<path:filename>')
def download(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

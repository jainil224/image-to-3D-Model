import io
import os
import sys
import unittest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import create_app


class AppTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = self.app.test_client()

    def test_health(self):
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['status'], 'ok')

    def test_upload_image(self):
        image_bytes = io.BytesIO()
        Image.new('RGB', (64, 64), color='red').save(image_bytes, format='PNG')
        image_bytes.seek(0)

        response = self.client.post('/generate', data={'image': (image_bytes, 'sample.png')}, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        self.assertIn('download', response.get_data(as_text=True).lower())


if __name__ == '__main__':
    unittest.main()

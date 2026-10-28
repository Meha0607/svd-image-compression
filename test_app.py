import io
import os
import shutil
import tempfile
import time
import unittest
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

import app as server


def image_bytes(array):
    data = io.BytesIO()
    Image.fromarray(array.astype("uint8")).save(data, "PNG")
    data.seek(0)
    return data


class ApiWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.client = server.app.test_client()
        rows, cols = np.mgrid[:16, :20]
        self.image = np.dstack((rows * 12, cols * 10, (rows + cols) * 7))

    def upload(self):
        response = self.client.post("/api/upload", data={"file": (image_bytes(self.image), "sample.png")})
        self.assertEqual(response.status_code, 200, response.json)
        return response.json["image_id"]

    def test_complete_workflow(self):
        image_id = self.upload()
        home = self.client.get("/")
        self.assertEqual(home.status_code, 200)
        self.assertIn(b"SVD IMAGE LAB", home.data)
        matrix = self.client.get("/api/matrix-basics", query_string={"image_id": image_id})
        self.assertEqual(matrix.status_code, 200)
        self.assertIn("rref_block", matrix.json)
        compressed = self.client.post("/api/compress", json={"image_id": image_id, "mode": "auto", "method": "energy", "param": 95})
        self.assertEqual(compressed.status_code, 200, compressed.json)
        self.assertGreaterEqual(compressed.json["energy_percent"], 95)
        self.assertIn("ssim", compressed.json)
        self.assertIn("mse", compressed.json)
        self.assertIn("file_sizes", compressed.json)
        download = self.client.get(f"/api/download/{image_id}/compressed")
        self.assertEqual(download.status_code, 200)
        download.close()
        suggestion = self.client.post("/api/rank-suggestion", json={"image_id": image_id, "method": "energy", "param": 90})
        self.assertEqual(suggestion.status_code, 200, suggestion.json)
        manual_compressed = self.client.post("/api/compress", json={"image_id": image_id, "mode": "manual", "k": 3})
        self.assertEqual(manual_compressed.status_code, 200, manual_compressed.json)
        noisy = self.client.post("/api/noise", json={"image_id": image_id, "kind": "saltpepper", "level": 10})
        self.assertEqual(noisy.status_code, 200, noisy.json)
        denoised = self.client.post("/api/denoise", json={"image_id": image_id, "mode": "auto"})
        self.assertEqual(denoised.status_code, 200, denoised.json)
        gaussian = self.client.post("/api/noise", json={"image_id": image_id, "kind": "gaussian", "level": 10})
        self.assertEqual(gaussian.status_code, 200, gaussian.json)
        manual_denoised = self.client.post("/api/denoise", json={"image_id": image_id, "mode": "manual", "k": 3})
        self.assertEqual(manual_denoised.status_code, 200, manual_denoised.json)
        self.assertIn("denoised_metrics", manual_denoised.json)
        download = self.client.get(f"/api/download/{image_id}/denoised")
        self.assertEqual(download.status_code, 200)
        download.close()

    def test_invalid_requests(self):
        self.assertEqual(self.client.post("/api/upload", data={}).status_code, 400)
        self.assertEqual(self.client.post("/api/compress", json={"image_id": "missing", "mode": "manual", "k": 1}).status_code, 400)
        self.assertEqual(self.client.post("/api/compress", json={"image_id": str(uuid.uuid4()), "mode": "manual", "k": 1}).status_code, 404)
        image_id = self.upload()
        self.assertEqual(self.client.post("/api/noise", json={"image_id": image_id, "kind": "bad", "level": 2}).status_code, 400)
        self.assertEqual(self.client.post("/api/compress", json={"image_id": image_id, "mode": "manual", "k": 1.5}).status_code, 400)
        self.assertEqual(self.client.post("/api/denoise", json={"image_id": image_id, "mode": "manual", "k": 1}).status_code, 404)
        self.assertEqual(self.client.post("/api/upload", data={"file": (io.BytesIO(b"not an image"), "bad.png")}).status_code, 400)
        self.assertEqual(self.client.post("/api/upload", data={"file": (image_bytes(self.image), "bad.gif")}).status_code, 400)

    def test_upload_limit(self):
        old_limit = server.app.config["MAX_CONTENT_LENGTH"]
        server.app.config["MAX_CONTENT_LENGTH"] = 1
        try:
            response = self.client.post("/api/upload", data={"file": (image_bytes(self.image), "sample.png")})
            self.assertEqual(response.status_code, 413)
        finally:
            server.app.config["MAX_CONTENT_LENGTH"] = old_limit

    def test_single_pixel_image_is_supported(self):
        pixel = np.array([[[12, 34, 56]]], dtype=np.uint8)
        response = self.client.post("/api/upload", data={"file": (image_bytes(pixel), "pixel.png")})
        self.assertEqual(response.status_code, 200, response.json)


class StorageCleanupTests(unittest.TestCase):
    def test_cleanup_removes_only_expired_inactive_uuid_folders(self):
        old_data_dir, old_retention = server.DATA_DIR, server.RETENTION_HOURS
        temporary = tempfile.mkdtemp(dir=old_data_dir)
        try:
            server.DATA_DIR, server.RETENTION_HOURS = Path(temporary).resolve(), 1
            old_id, current_id, unrelated = str(uuid.uuid4()), str(uuid.uuid4()), "notes"
            for folder in (old_id, current_id, unrelated):
                (server.DATA_DIR / folder).mkdir()
            old_time = time.time() - 7200
            os.utime(server.DATA_DIR / old_id, (old_time, old_time))
            os.utime(server.DATA_DIR / unrelated, (old_time, old_time))
            self.assertEqual(server.cleanup_expired_data(), 1)
            self.assertFalse((server.DATA_DIR / old_id).exists())
            self.assertTrue((server.DATA_DIR / current_id).exists())
            self.assertTrue((server.DATA_DIR / unrelated).exists())
        finally:
            server.DATA_DIR, server.RETENTION_HOURS = old_data_dir, old_retention
            shutil.rmtree(temporary, ignore_errors=True)


import unittest
import numpy as np

import compression
from svd_core import decompose, gram_schmidt, null_space_basis, reconstruct, rref


class SvdCoreTests(unittest.TestCase):
    def test_full_rank_reconstruction_gray_and_rgb(self):
        gray = np.arange(30, dtype=np.uint8).reshape(5, 6)
        rgb = np.dstack((gray, 255 - gray, gray // 2))
        self.assertTrue(np.array_equal(reconstruct(decompose(gray), 5), gray))
        self.assertTrue(np.array_equal(reconstruct(decompose(rgb), 5), rgb))

    def test_matrix_operations(self):
        matrix = np.array([[1., 2., 3.], [2., 4., 6.]])
        reduced, pivots = rref(matrix)
        self.assertEqual(pivots, [0])
        basis = null_space_basis(matrix)
        self.assertTrue(np.allclose(matrix @ basis.T, 0))
        q = gram_schmidt(matrix)
        self.assertTrue(np.allclose(q @ q.T, np.eye(len(q))))

    def test_zero_energy_auto_k(self):
        svd = decompose(np.zeros((4, 4), dtype=np.uint8))
        self.assertEqual(compression.auto_k_energy(svd)["k"], 1)

    def test_rank_must_be_an_integer(self):
        svd = decompose(np.eye(4, dtype=np.uint8))
        with self.assertRaises(ValueError):
            reconstruct(svd, 1.5)
        with self.assertRaises(ValueError):
            compression.compress(svd, np.eye(4, dtype=np.uint8), float("nan"))

    def test_compression_metrics_for_rgb(self):
        image = np.full((5, 6, 3), 120, dtype=np.uint8)
        reconstructed, info = compression.compress(decompose(image), image, 1)
        self.assertTrue(np.array_equal(reconstructed, image))
        self.assertEqual(info["energy"], 100.0)
        self.assertEqual(info["psnr"], 100.0)

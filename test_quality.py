import unittest

import numpy as np

from quality import metrics, mse, psnr, ssim


class QualityMetricTests(unittest.TestCase):
    def test_identical_images_have_perfect_metrics(self):
        image = np.full((8, 8, 3), 120, dtype=np.uint8)
        values = metrics(image, image)
        self.assertEqual(values["mse"], 0.0)
        self.assertEqual(values["psnr"], 100.0)
        self.assertEqual(values["ssim"], 1.0)

    def test_metrics_reject_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            mse(np.zeros((2, 2)), np.zeros((3, 3)))

    def test_small_and_changed_images_are_finite(self):
        source = np.array([[0]], dtype=np.uint8)
        comparison = np.array([[255]], dtype=np.uint8)
        self.assertEqual(mse(source, comparison), 65025.0)
        self.assertTrue(np.isfinite(psnr(source, comparison)))
        self.assertTrue(np.isfinite(ssim(source, comparison)))

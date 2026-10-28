"""
test_denoise.py - my own test for denoise.py (not part of the team's final app)

Put any picture named sample.jpg (or sample.png) in this folder, then run:
    python test_denoise.py
If no picture is found, a simple made-up image is used instead.
"""

import os
import unittest
import numpy as np
from PIL import Image
from denoise import add_noise, auto_k_denoise, auto_k_denoise_image, best_k_by_psnr, denoise_with_k, psnr
from svd_core import decompose


class DenoisingTests(unittest.TestCase):
    def test_noise_validation_and_shape(self):
        image = np.full((8, 9, 3), 127, dtype=np.uint8)
        np.random.seed(0)
        noisy = add_noise(image, "gaussian", 10)
        self.assertEqual(noisy.shape, image.shape)
        self.assertGreater(np.abs(noisy - image).mean(), 0)
        with self.assertRaises(ValueError):
            add_noise(image, "saltpepper", 1.1)
        with self.assertRaises(ValueError):
            add_noise(image, "gaussian", -1)

    def test_auto_rank_supports_rgb_svd(self):
        image = np.full((8, 9, 3), 127, dtype=np.uint8)
        k, reason = auto_k_denoise_image(decompose(image), image.shape[:2])
        self.assertEqual(k, 1)
        self.assertIn("Automatic selection", reason)


# ---------- Step 1: load an image as a grayscale matrix ----------
def load_image():
    for name in ["sample.jpg", "sample.jpeg", "sample.png"]:
        if os.path.exists(name):
            img = Image.open(name).convert("L")      # "L" = grayscale
            img.thumbnail((256, 256))                # shrink so the test is fast
            print("Loaded:", name, "->", img.size)
            return np.array(img, dtype=float)
    print("No sample image found, using a made-up image.")
    x = np.linspace(0, 255, 200)
    A = np.add.outer(x, x) / 2                       # smooth gradient
    A[50:150, 60:140] = 220                          # a bright rectangle
    A[20:60, 130:180] = 40                           # a dark rectangle
    return A


def main():
    """Optional visual experiment; kept out of automated test discovery."""
    import matplotlib.pyplot as plt

    A = load_image()
    print("Image shape:", A.shape)
    np.random.seed(1)
    tests = [("gaussian", 10), ("gaussian", 25), ("gaussian", 50),
             ("saltpepper", 0.02), ("saltpepper", 0.05)]
    print("\nkind        level | PSNR noisy | PSNR auto k | PSNR best k | auto k | best k")
    print("-" * 80)
    saved = None
    for kind, level in tests:
        A_noisy = add_noise(A, kind, level)
        U, S, Vt = np.linalg.svd(A_noisy, full_matrices=False)
        k, reason = auto_k_denoise(S, A.shape)
        A_den = denoise_with_k(U, S, Vt, k)
        k_best = best_k_by_psnr(A, U, S, Vt)
        p_noisy, p_den = psnr(A, A_noisy), psnr(A, A_den)
        p_best = psnr(A, denoise_with_k(U, S, Vt, k_best))
        print(f"{kind:<11} {level:<5} | {p_noisy:10.2f} | {p_den:11.2f} | {p_best:11.2f} | {k:6d} | {k_best:6d}")
        if kind == "gaussian" and level == 25:
            saved = (A_noisy, A_den, k, p_noisy, p_den, reason)
    A_noisy, A_den, k, p_noisy, p_den, reason = saved
    print("\nReason text:", reason)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(A, cmap="gray", vmin=0, vmax=255); axes[0].set_title("Original")
    axes[1].imshow(A_noisy, cmap="gray", vmin=0, vmax=255); axes[1].set_title(f"Noisy (PSNR {p_noisy:.1f} dB)")
    axes[2].imshow(A_den, cmap="gray", vmin=0, vmax=255); axes[2].set_title(f"Denoised k={k} (PSNR {p_den:.1f} dB)")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("result.png", dpi=120)
    print("Saved picture: result.png")
    plt.show()


if __name__ == "__main__":
    main()


"""
denoise.py  -  Noise reduction and automatic k selection (Person 4)

Functions:
    add_noise(A, kind, level)          -> A_noisy
    psnr(A, B)                         -> float (higher = better)
    denoise_with_k(U, S, Vt, k)        -> A_denoised   (small helper)
    auto_k_denoise(S, shape)           -> k, reason_text
    best_k_by_psnr(A_original, U, S, Vt) -> k          (benchmark mode)

Images are NumPy arrays with values 0-255 (grayscale = one 2D matrix).
For color images the caller runs these functions once per R, G, B matrix.
"""

import numpy as np
from PIL import Image, ImageFilter

# Use the team's reconstruct() from svd_core.py if it exists.
# If svd_core.py is not ready yet, use a local copy with the same behaviour,
# so this file can still be tested on its own.
try:
    from svd_core import reconstruct
except ImportError:
    def reconstruct(U, S, Vt, k):
        # A_k = (first k columns of U) * (first k singular values) * (first k rows of Vt)
        return (U[:, :k] * S[:k]) @ Vt[:k, :]


# ---------------------------------------------------------------
# 1. Adding noise (to create a noisy test image)
# ---------------------------------------------------------------
def add_noise(A, kind, level):
    """
    kind = "gaussian"  : add random values with standard deviation = level
                         (level example: 10, 25, 50)
    kind = "saltpepper": turn a fraction 'level' of pixels into 0 or 255
                         (level example: 0.05 means 5% of pixels)
    Returns a float array clipped to 0-255.
    """
    A = np.asarray(A, dtype=float)
    if isinstance(level, bool):
        raise ValueError("level must be a number")
    try:
        level = float(level)
    except (TypeError, ValueError) as exc:
        raise ValueError("level must be a number") from exc
    if not np.isfinite(level):
        raise ValueError("level must be finite")

    if kind == "gaussian":
        if level < 0:
            raise ValueError("Gaussian noise level must be non-negative")
        noise = np.random.normal(0, level, A.shape)   # mean 0, std = level
        A_noisy = A + noise

    elif kind == "saltpepper":
        if not 0 <= level <= 1:
            raise ValueError("Salt-and-pepper noise level must be between 0 and 1")
        A_noisy = A.copy()
        r = np.random.random(A.shape)        # one random number per pixel
        hit = r < level                      # these pixels get damaged
        coin = np.random.random(A.shape) < 0.5   # half black, half white
        A_noisy[hit & coin] = 0              # pepper (black)
        A_noisy[hit & ~coin] = 255           # salt (white)

    else:
        raise ValueError("kind must be 'gaussian' or 'saltpepper'")

    return np.clip(A_noisy, 0, 255)          # keep valid pixel range


# ---------------------------------------------------------------
# 2. PSNR helper (quality measure)
# ---------------------------------------------------------------
def psnr(A, B):
    """
    Peak Signal-to-Noise Ratio in dB. Higher means B is closer to A.
    PSNR = 10 * log10(255^2 / MSE), MSE = mean of squared differences.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    mse = np.mean((A - B) ** 2)
    if mse == 0:
        return 100.0     # identical images (100 instead of infinity, so JSON works)
    return float(10 * np.log10(255.0 ** 2 / mse))


# ---------------------------------------------------------------
# 3. Denoising
# ---------------------------------------------------------------
def denoise_with_k(U, S, Vt, k):
    """
    Denoise = keep only the top k singular values ("layers") of the NOISY image.

    Why this works (simple words):
      - A = sum of layers: s1*u1*v1^T + s2*u2*v2^T + ...
      - The first layers are big and carry the real picture (edges, shapes).
      - The last layers are tiny and mostly carry random noise.
      - Keeping the top k layers gives A_k, which is the "least squares
        best rank-k fit" of the noisy image (Eckart-Young theorem):
        no other rank-k matrix is closer to it in squared error.
      - Throwing away the small layers removes much of the noise.
    """
    A_k = reconstruct(U, S, Vt, k)
    return np.clip(A_k, 0, 255)


# ---------------------------------------------------------------
# 4. Automatic k  (Gavish-Donoho rule)
# ---------------------------------------------------------------
def auto_k_denoise(S, shape):
    """
    Pick k automatically using the Gavish-Donoho optimal hard threshold.
        cutoff = omega(beta) * median(S)
        beta   = min(m, n) / max(m, n)
        omega(beta) = 0.56*beta^3 - 0.95*beta^2 + 1.82*beta + 1.43
    For a square matrix beta = 1, so omega is about 2.858.
    k = number of singular values bigger than the cutoff.
    """
    S = np.asarray(S, dtype=float)
    m, n = shape
    if m < 1 or n < 1:
        raise ValueError("image shape must be positive")
    beta = min(m, n) / max(m, n)
    omega = 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43
    # A mathematical rank-one matrix can still acquire tiny singular values
    # from floating-point SVD arithmetic.  Do not treat those as signal when
    # the median-based threshold is itself close to zero.
    numerical_floor = np.finfo(float).eps * max(m, n) * (float(S[0]) if S.size else 0.0)
    cutoff = max(omega * np.median(S), numerical_floor)

    k = int(np.sum(S > cutoff))
    # Do not force a 95%-energy rank: that makes the estimated noise tail
    # nearly empty and results in almost no denoising on real photos.
    k = min(max(k, 1), len(S))

    reason = (f"Adaptive SVD threshold detected {k} dominant components; "
              "smaller components are smoothly reduced to retain detail.")
    return k, reason


def median_filter_image(image, size=3):
    """Remove isolated salt-and-pepper pixels before the SVD step."""
    array = np.asarray(image, dtype=np.uint8)
    if size < 3 or size % 2 == 0:
        raise ValueError("Median filter size must be an odd number of at least 3")
    return np.asarray(Image.fromarray(array).filter(ImageFilter.MedianFilter(size=size)))


def denoise_reconstruct(svd, k):
    """Adaptive soft SVD shrinkage that suppresses noise without hard blur."""
    parts = svd if isinstance(svd, list) else [svd]
    output = []
    for U, S, Vt in parts:
        k_channel = min(max(int(k), 1), len(S))
        lower_spectrum = S[max(k_channel, len(S) // 2):]
        noise_floor = float(np.median(lower_spectrum)) if lower_spectrum.size else float(np.median(S))
        # Quadratic adaptive shrinkage is stronger on weak/noise components
        # but keeps high-energy structures and edges close to intact.
        threshold = 1.5 * noise_floor
        filtered = S * np.maximum(1.0 - (threshold / np.maximum(S, 1e-12)) ** 2, 0.0)
        output.append((U * filtered) @ Vt)
    image = np.stack(output, axis=2) if isinstance(svd, list) else output[0]
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def auto_k_denoise_image(svd, shape):
    """Select one valid rank for a grayscale or RGB SVD decomposition.

    RGB channels are reconstructed with a common rank so the result keeps a
    straightforward rank-k interface everywhere in the application.
    """
    parts = svd if isinstance(svd, list) else [svd]
    choices = [auto_k_denoise(S, shape) for _, S, _ in parts]
    k = max(choice[0] for choice in choices)
    return k, "Automatic selection: " + choices[0][1]


# ---------------------------------------------------------------
# 5. Benchmark mode: best k using the ORIGINAL (clean) image
# ---------------------------------------------------------------
def best_k_by_psnr(A_original, U, S, Vt):
    """
    Try k values on the SVD of the noisy image (U, S, Vt) and return the
    k whose result has the highest PSNR against the clean original.
    Only possible when we know the original, so it is a benchmark to
    check how good auto_k_denoise is.
    """
    r = len(S)
    step = 1 if r <= 300 else r // 60          # small images: test every k
    best_k, best_psnr = 1, -1.0
    for k in range(1, r + 1, step):
        score = psnr(A_original, denoise_with_k(U, S, Vt, k))
        if score > best_psnr:
            best_k, best_psnr = k, score
    return best_k


# ---------------------------------------------------------------
# Small test: run with   python denoise.py
# ---------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)                          # same result every run

    # 1) Make a clean 100x100 matrix that has low rank (rank about 5)
    X = np.random.randn(100, 5)
    Y = np.random.randn(5, 100)
    A = X @ Y
    A = (A - A.min()) / (A.max() - A.min()) * 255   # scale to 0-255

    # 2) Add noise
    A_noisy = add_noise(A, "gaussian", 25)

    # 3) SVD of the noisy matrix
    U, S, Vt = np.linalg.svd(A_noisy, full_matrices=False)

    # 4) Auto k and denoise
    k, reason = auto_k_denoise(S, A.shape)
    A_den = denoise_with_k(U, S, Vt, k)

    # 5) Benchmark k
    k_best = best_k_by_psnr(A, U, S, Vt)

    print(reason)
    print("PSNR noisy    vs original:", round(psnr(A, A_noisy), 2), "dB")
    print("PSNR denoised vs original:", round(psnr(A, A_den), 2), "dB")
    print("Auto k =", k, "| best k by PSNR =", k_best)
    print("Denoising improved PSNR:", psnr(A, A_den) > psnr(A, A_noisy))

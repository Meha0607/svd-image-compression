"""
compression.py  (Person 3)
==========================
Image compression using truncated SVD (low-rank approximation).

SVD INPUT FORMAT (agree on this with Person 2):
  * Grayscale image : svd = (U, S, Vt)
  * Color image     : svd = [(U_r, S_r, Vt_r), (U_g, S_g, Vt_g), (U_b, S_b, Vt_b)]
  The original image is a numpy array, values 0-255,
  shape (m, n) for grayscale or (m, n, 3) for color.

Main function for the server:  compress(svd, original, k)
Auto-k functions each return {"k": int, "reason": str}.
"""

import numpy as np

from svd_core import clamp_rank, decompose as compute_svd, reconstruct as reconstruct_image


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _is_color(svd):
    """Color SVD is a list of 3 (U, S, Vt) tuples; grayscale is one tuple."""
    return isinstance(svd, list)


def _channels(svd):
    return svd if _is_color(svd) else [svd]


def _shape(svd):
    """Return (m, n, r) where r = max possible rank = min(m, n)."""
    U, S, Vt = _channels(svd)[0]
    return U.shape[0], Vt.shape[1], len(S)


def combined_singular_values(svd):
    """
    One singular-value curve for the whole image.
    For color we combine channels as sqrt(Sr^2 + Sg^2 + Sb^2), so the
    energy calculation stays exact. For grayscale it is just S.
    """
    Ss = [np.asarray(S, dtype=float) for (_, S, _) in _channels(svd)]
    return np.sqrt(sum(s ** 2 for s in Ss))


def _clamp_k(k, r):
    return clamp_rank(k, r)


# ----------------------------------------------------------------------
# 1. Rank-k reconstruction
# ----------------------------------------------------------------------
def rank_k_matrix(U, S, Vt, k):
    """A_k = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]  (one matrix / one channel)."""
    return (U[:, :k] * S[:k]) @ Vt[:k, :]   # same as U_k @ diag(S_k) @ Vt_k


def reconstruct(svd, k):
    """
    Rank-k reconstruction of the whole image.
    Color: reconstruct each channel, stack, then clip to 0-255.
    Returns a uint8 array.
    """
    return reconstruct_image(svd, _clamp_k(k, _shape(svd)[2]))


# ----------------------------------------------------------------------
# 2. Metrics
# ----------------------------------------------------------------------
def compression_ratio(m, n, k):
    """
    Original storage : m*n numbers
    Rank-k storage   : k*(m + n + 1)   (k columns of U, k rows of Vt, k sigmas)
    ratio < 1 means the "compressed" version is actually BIGGER.
    (For color both sides are multiplied by 3, so the ratio is unchanged.)
    """
    return (m * n) / (k * (m + n + 1))


def energy_percent(S, k):
    """Share of total energy (sum of sigma^2) kept by the first k values."""
    S = np.asarray(S, dtype=float)
    total = np.sum(S ** 2)
    if total == 0:
        return 100.0
    return float(np.sum(S[:k] ** 2) / total * 100)


def psnr(original, reconstructed, cap=100.0):
    """PSNR in dB for 8-bit images. Perfect match -> returns `cap`."""
    diff = original.astype(np.float64) - reconstructed.astype(np.float64)
    mse = np.mean(diff ** 2)
    if mse == 0:
        return cap
    return float(min(cap, 10 * np.log10(255.0 ** 2 / mse)))


# ----------------------------------------------------------------------
# 3. Auto-k (each returns {"k": ..., "reason": ...})
# ----------------------------------------------------------------------
def auto_k_energy(svd, threshold=95.0):
    """Smallest k whose energy >= threshold %."""
    S = combined_singular_values(svd)
    total = np.sum(S ** 2)
    if total == 0:
        return {"k": 1, "reason": "The image has zero energy; using the minimum rank k=1."}
    cum = np.cumsum(S ** 2) / total * 100
    idx = int(np.searchsorted(cum, threshold))   # first index with cum >= threshold
    k = _clamp_k(idx + 1, len(S))
    return {"k": k,
            "reason": f"k={k} keeps {cum[k-1]:.1f}% of the image's energy "
                      f"(target was {threshold:g}%)."}


def auto_k_ratio(svd, target_ratio=10.0):
    """k that gives roughly the requested compression ratio."""
    m, n, r = _shape(svd)
    k = _clamp_k(np.floor(m * n / (target_ratio * (m + n + 1))), r)
    actual = compression_ratio(m, n, k)
    return {"k": k,
            "reason": f"k={k} gives about {actual:.1f}x compression "
                      f"(target was {target_ratio:g}x)."}


def auto_k_elbow(svd):
    """
    Kneedle-style elbow: after normalising the singular-value curve to
    [0, 1] on both axes, pick the point farthest from the straight line
    joining the first and last points.
    """
    S = combined_singular_values(svd)
    r = len(S)
    if r < 3:
        return {"k": r, "reason": "Image too small for an elbow; using all layers."}

    x = np.linspace(0, 1, r)
    span = S[0] - S[-1]
    y = (S - S[-1]) / span if span > 0 else np.zeros(r)

    # distance of each point from line (0,1) -> (1,0):  x + y - 1 = 0
    dist = np.abs(x + y - 1) / np.sqrt(2)
    k = _clamp_k(int(np.argmax(dist)) + 1, r)
    return {"k": k,
            "reason": f"Singular values flatten after k={k}, "
                      f"so extra layers add little."}


def auto_k(svd, method="energy", **kwargs):
    """Single entry point for the server: method = energy | ratio | elbow."""
    methods = {"energy": auto_k_energy, "ratio": auto_k_ratio, "elbow": auto_k_elbow}
    if method not in methods:
        raise ValueError(f"Unknown method '{method}'. Use: {list(methods)}")
    return methods[method](svd, **kwargs)


# ----------------------------------------------------------------------
# 4. Summary function for the server
# ----------------------------------------------------------------------
def compress(svd, original, k):
    """
    Returns (reconstructed_uint8_image, info_dict).
    info = {"k", "psnr", "ratio", "energy"}
    """
    m, n, r = _shape(svd)
    k = _clamp_k(k, r)
    recon = reconstruct(svd, k)
    S = combined_singular_values(svd)
    info = {
        "k": k,
        "psnr": round(psnr(original, recon), 2),
        "ratio": round(compression_ratio(m, n, k), 2),
        "energy": round(energy_percent(S, k), 2),
    }
    return recon, info


# ----------------------------------------------------------------------
# 5. Table + graph for the PPT
# ----------------------------------------------------------------------
def results_table(img, ks=(5, 10, 20, 50, 100)):
    """List of dicts: k, psnr, ratio, energy (for the PPT table)."""
    svd = compute_svd(img)
    _, _, r = _shape(svd)
    rows = []
    for k in ks:
        if k > r:
            continue
        _, info = compress(svd, img, k)
        rows.append(info)
    return rows


def print_table(rows, title=""):
    if title:
        print(f"\n{title}")
    print("| k | PSNR (dB) | Compression ratio | Energy % |")
    print("|---|-----------|-------------------|----------|")
    for r in rows:
        print(f"| {r['k']} | {r['psnr']} | {r['ratio']} | {r['energy']} |")


def plot_results(rows, path="k_vs_psnr_ratio.png", title="k vs PSNR vs Compression ratio"):
    """Twin-axis plot: PSNR (left) and compression ratio (right) against k."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ks = [r["k"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(ks, [r["psnr"] for r in rows], "o-", color="tab:blue", label="PSNR")
    ax1.set_xlabel("k (number of singular values kept)")
    ax1.set_ylabel("PSNR (dB)", color="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(ks, [r["ratio"] for r in rows], "s--", color="tab:red", label="Ratio")
    ax2.set_ylabel("Compression ratio (x)", color="tab:red")
    ax1.grid(alpha=0.3)
    plt.title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# Self-test:  python compression.py [image_path]
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        from PIL import Image
        im = Image.open(sys.argv[1]).convert("L")
        im.thumbnail((512, 512))
        img = np.array(im)
    else:
        # synthetic test image so the file runs without any input
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[0:256, 0:256]
        img = (127 + 60 * np.sin(xx / 20) * np.cos(yy / 30)
               + 40 * ((xx - 128) ** 2 + (yy - 128) ** 2 < 60 ** 2)
               + rng.normal(0, 3, (256, 256)))
        img = np.clip(img, 0, 255).astype(np.uint8)

    svd = compute_svd(img)
    m, n, r = _shape(svd)

    # sanity checks
    full, info = compress(svd, img, r)
    print("Full rank   :", info, "(PSNR should be very high)")
    rows = results_table(img)
    print_table(rows, "k vs PSNR vs ratio vs energy")
    assert all(rows[i]["psnr"] <= rows[i + 1]["psnr"] for i in range(len(rows) - 1))
    assert all(rows[i]["ratio"] >= rows[i + 1]["ratio"] for i in range(len(rows) - 1))

    print("\nAuto-k:")
    for res in (auto_k_energy(svd, 95), auto_k_energy(svd, 99),
                auto_k_ratio(svd, 10), auto_k_elbow(svd)):
        print(" ", res)

    print("Graph saved to:", plot_results(rows))

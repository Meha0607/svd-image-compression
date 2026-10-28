"""Shared numerical routines for the SVD Image Lab.

Images are uint8 grayscale ``(H, W)`` or RGB ``(H, W, 3)`` arrays.  An
image decomposition is one SVD tuple for grayscale or one tuple per channel
for RGB.  Keeping this representation small and explicit makes it usable by
both compression and denoising.
"""
from __future__ import annotations

import numpy as np


def decompose(image: np.ndarray):
    image = np.asarray(image)
    if image.ndim == 2:
        return np.linalg.svd(image.astype(float), full_matrices=False)
    if image.ndim == 3 and image.shape[2] == 3:
        return [np.linalg.svd(image[:, :, channel].astype(float), full_matrices=False)
                for channel in range(3)]
    raise ValueError("Expected a grayscale or RGB image.")


def rank_limit(svd) -> int:
    channels = svd if isinstance(svd, list) else [svd]
    return min(len(part[1]) for part in channels)


def clamp_rank(k, maximum: int) -> int:
    if isinstance(k, bool) or not isinstance(maximum, (int, np.integer)) or maximum < 1:
        raise ValueError("k must be an integer.")
    try:
        numeric_k = float(k)
    except (TypeError, ValueError) as exc:
        raise ValueError("k must be an integer.") from exc
    if not np.isfinite(numeric_k) or not numeric_k.is_integer():
        raise ValueError("k must be an integer.")
    k = int(numeric_k)
    return max(1, min(k, maximum))


def reconstruct(*args):
    """Reconstruct an image decomposition or one ``U, S, Vt, k`` channel.

    The four-argument form intentionally remains compatible with the existing
    denoising module; the two-argument form is the shared image API.
    """
    if len(args) == 4:
        U, S, Vt, k = args
        k = clamp_rank(k, len(S))
        return (U[:, :k] * S[:k]) @ Vt[:k, :]
    if len(args) != 2:
        raise TypeError("reconstruct expects (svd, k) or (U, S, Vt, k)")
    svd, k = args
    k = clamp_rank(k, rank_limit(svd))
    parts = svd if isinstance(svd, list) else [svd]
    result = [reconstruct(U, S, Vt, k) for U, S, Vt in parts]
    image = np.stack(result, axis=2) if isinstance(svd, list) else result[0]
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def grayscale(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    if image.ndim == 2:
        return image
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def rref(matrix, tolerance=1e-10):
    """Return reduced row-echelon form and pivot columns using stable pivots."""
    a = np.asarray(matrix, dtype=float).copy()
    rows, columns = a.shape
    pivots, pivot_row = [], 0
    scale = max(1.0, float(np.max(np.abs(a))))
    eps = tolerance * scale
    for column in range(columns):
        candidate = pivot_row + int(np.argmax(np.abs(a[pivot_row:, column]))) if pivot_row < rows else rows
        if candidate >= rows or abs(a[candidate, column]) <= eps:
            continue
        a[[pivot_row, candidate]] = a[[candidate, pivot_row]]
        a[pivot_row] /= a[pivot_row, column]
        for row in range(rows):
            if row != pivot_row:
                a[row] -= a[row, column] * a[pivot_row]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == rows:
            break
    a[np.abs(a) <= eps] = 0
    return a, pivots


def null_space_basis(matrix):
    reduced, pivots = rref(matrix)
    cols = reduced.shape[1]
    free = [column for column in range(cols) if column not in pivots]
    basis = []
    for free_column in free:
        vector = np.zeros(cols)
        vector[free_column] = 1.0
        for row, pivot in enumerate(pivots):
            vector[pivot] = -reduced[row, free_column]
        basis.append(vector)
    return np.asarray(basis)


def gram_schmidt(vectors, tolerance=1e-10):
    """Return an orthonormal basis, dropping linearly dependent vectors."""
    result = []
    for vector in np.asarray(vectors, dtype=float):
        q = vector.copy()
        for basis_vector in result:
            q -= np.dot(q, basis_vector) * basis_vector
        norm = np.linalg.norm(q)
        if norm > tolerance:
            result.append(q / norm)
    return np.asarray(result)

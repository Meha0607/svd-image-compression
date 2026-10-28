"""Flask API and UI entry point for SVD Image Lab."""
from __future__ import annotations

import base64
import io
import json
import os
import shutil
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from PIL import Image, UnidentifiedImageError

import compression
import denoise
from quality import metrics
from svd_core import decompose, gram_schmidt, grayscale, null_space_basis, rank_limit, rref

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("SVD_DATA_DIR", ROOT / "data")).resolve()
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
MAX_SIDE = int(os.getenv("SVD_MAX_SIDE", "1024"))
MAX_UPLOAD_BYTES = int(os.getenv("SVD_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
RETENTION_HOURS = float(os.getenv("SVD_RETENTION_HOURS", "24"))
METADATA_NAME = "metadata.json"
_active_ids: set[str] = set()
_active_lock = threading.Lock()
app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES)


def error(message, code=400):
    return jsonify(error=message), code


def image_path(image_id, kind):
    if kind not in {"original", "compressed", "noisy", "denoised"}:
        raise ValueError("Invalid image type.")
    try:
        image_id = str(uuid.UUID(str(image_id)))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid image ID.") from exc
    return DATA_DIR / image_id / f"{kind}.png"


def metadata_path(image_id):
    return image_path(image_id, "original").parent / METADATA_NAME


def write_metadata(image_id, source_name, original_bytes):
    path = metadata_path(image_id)
    payload = {"source_name": Path(source_name).name, "original_bytes": int(original_bytes),
               "created_at": datetime.now(timezone.utc).isoformat()}
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload), encoding="utf-8")
    temp.replace(path)


def read_metadata(image_id):
    try:
        return json.loads(metadata_path(image_id).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise FileNotFoundError("Image metadata was not found. Upload the image again.") from exc


def file_size_summary(image_id, kind):
    metadata = read_metadata(image_id)
    path = image_path(image_id, kind)
    if not path.is_file():
        raise FileNotFoundError("Processed image not found. Process the image first.")
    original_bytes, processed_bytes = int(metadata["original_bytes"]), path.stat().st_size
    saved = original_bytes - processed_bytes
    return {"original_bytes": original_bytes, "processed_bytes": processed_bytes,
            "bytes_saved": saved,
            "percent_saved": round((saved / original_bytes * 100) if original_bytes else 0, 2)}


@contextmanager
def active_image(image_id):
    parsed = str(uuid.UUID(str(image_id)))
    with _active_lock:
        _active_ids.add(parsed)
    folder = DATA_DIR / parsed
    if folder.is_dir():
        try:
            folder.touch()
        except OSError:
            pass
    try:
        yield parsed
    finally:
        with _active_lock:
            _active_ids.discard(parsed)


def cleanup_expired_data(now=None):
    """Remove only expired UUID upload folders not being processed by this app."""
    if RETENTION_HOURS < 0 or not DATA_DIR.exists():
        return 0
    now = now or datetime.now(timezone.utc).timestamp()
    cutoff = now - RETENTION_HOURS * 3600
    removed = 0
    for folder in DATA_DIR.iterdir():
        if not folder.is_dir():
            continue
        try:
            image_id = str(uuid.UUID(folder.name))
            if folder.resolve().parent != DATA_DIR or folder.stat().st_mtime >= cutoff:
                continue
            with _active_lock:
                if image_id in _active_ids:
                    continue
            shutil.rmtree(folder)
            removed += 1
        except (ValueError, OSError):
            # Ignore unrelated/non-UUID folders and recoverable filesystem races.
            continue
    return removed


def read_image(image_id, kind="original"):
    path = image_path(image_id, kind)
    if not path.is_file():
        raise FileNotFoundError("Image not found. Upload it again.")
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB"))


def save_image(image_id, kind, image):
    path = image_path(image_id, kind)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(path, "PNG")
    return path


def as_base64(image):
    output = io.BytesIO()
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(output, "PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def number(payload, name, minimum=None, maximum=None):
    value = payload.get(name)
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number.")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if not np.isfinite(value) or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside the supported range.")
    return value


def integer(payload, name, minimum, maximum):
    value = number(payload, name, minimum, maximum)
    if not value.is_integer():
        raise ValueError(f"{name} must be an integer.")
    return int(value)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/upload")
def upload():
    file = request.files.get("file")
    if not file or not file.filename:
        return error("Choose an image file to upload.")
    extension = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if extension not in ALLOWED_EXTENSIONS:
        return error("Only PNG, JPG, JPEG, and BMP images are supported.")
    try:
        raw_upload = file.read()
        with Image.open(io.BytesIO(raw_upload)) as source:
            source.load()
            if source.width < 1 or source.height < 1:
                return error("Image dimensions must be positive.")
            if source.width > MAX_SIDE or source.height > MAX_SIDE:
                return error(f"Images must be at most {MAX_SIDE} pixels on either side for SVD processing.")
            image = np.asarray(source.convert("RGB"))
    except (UnidentifiedImageError, OSError):
        return error("The uploaded file is not a readable image.")
    image_id = str(uuid.uuid4())
    cleanup_expired_data()
    with active_image(image_id):
        save_image(image_id, "original", image)
        write_metadata(image_id, file.filename, len(raw_upload))
    singular_values = compression.combined_singular_values(decompose(image))
    return jsonify(image_id=image_id, shape=list(image.shape[:2]), singular_values=np.round(singular_values, 6).tolist(),
                   original_bytes=len(raw_upload))


@app.get("/api/matrix-basics")
def matrix_basics():
    try:
        image = read_image(request.args.get("image_id"))
        block = grayscale(image)[:6, :6]
        reduced, pivots = rref(block)
        basis = null_space_basis(block)
        orthogonal = gram_schmidt(block)
        return jsonify(rank=len(pivots), nullity=block.shape[1] - len(pivots),
                       rref_block=np.round(reduced, 6).tolist(),
                       basis=np.round(basis, 6).tolist(),
                       gram_schmidt=np.round(orthogonal, 6).tolist())
    except (ValueError, FileNotFoundError) as exc:
        return error(str(exc), 404 if isinstance(exc, FileNotFoundError) else 400)


@app.post("/api/rank-suggestion")
def rank_suggestion():
    """Preview an automatic compression rank without writing a result file."""
    payload = request.get_json(silent=True) or {}
    try:
        with active_image(payload.get("image_id")) as image_id:
            image = read_image(image_id)
            svd = decompose(image)
            method = payload.get("method", "energy")
            kwargs = {"threshold": number(payload, "param", 1, 100)} if method == "energy" else ({"target_ratio": number(payload, "param", 0.01)} if method == "ratio" else {})
            choice = compression.auto_k(svd, method, **kwargs)
            _, info = compression.compress(svd, image, choice["k"])
            return jsonify(k_used=choice["k"], reason=choice["reason"],
                           compression_ratio=info["ratio"], energy_percent=info["energy"])
    except (ValueError, FileNotFoundError) as exc:
        return error(str(exc), 404 if isinstance(exc, FileNotFoundError) else 400)


@app.post("/api/compress")
def compress_image():
    payload = request.get_json(silent=True) or {}
    try:
        with active_image(payload.get("image_id")) as image_id:
            original = read_image(image_id)
            svd = decompose(original)
            if payload.get("mode") == "auto":
                method = payload.get("method", "energy")
                kwargs = {"threshold": number(payload, "param", 1, 100)} if method == "energy" else ({"target_ratio": number(payload, "param", 0.01)} if method == "ratio" else {})
                choice = compression.auto_k(svd, method, **kwargs)
                k, reason = choice["k"], choice["reason"]
            elif payload.get("mode") == "manual":
                k = integer(payload, "k", 1, rank_limit(svd))
                reason = f"Manual choice: using k = {k}."
            else:
                raise ValueError("mode must be 'manual' or 'auto'.")
            result, info = compression.compress(svd, original, k)
            save_image(image_id, "compressed", result)
            values = metrics(original, result)
            return jsonify(k_used=info["k"], reason=reason, image=as_base64(result),
                           compression_ratio=info["ratio"], energy_percent=info["energy"],
                           **values, file_sizes=file_size_summary(image_id, "compressed"))
    except (ValueError, FileNotFoundError) as exc:
        return error(str(exc), 404 if isinstance(exc, FileNotFoundError) else 400)


@app.get("/api/download/<image_id>/<kind>")
def download_image(image_id, kind):
    try:
        path = image_path(image_id, kind)
        if not path.is_file() or kind == "original":
            raise FileNotFoundError("Processed image not found. Process the image first.")
        return send_file(path, mimetype="image/png", as_attachment=True,
                         download_name=f"svd-{kind}-{image_id}.png")
    except (ValueError, FileNotFoundError) as exc:
        return error(str(exc), 404 if isinstance(exc, FileNotFoundError) else 400)


@app.post("/api/noise")
def add_noise():
    payload = request.get_json(silent=True) or {}
    try:
        with active_image(payload.get("image_id")) as image_id:
            original = read_image(image_id)
            kind = payload.get("kind")
            raw_level = number(payload, "level", 0)
            if kind == "gaussian":
                level = number(payload, "level", 0, 100)
            elif kind == "saltpepper":
                level = number(payload, "level", 0, 100) / 100
            else:
                raise ValueError("kind must be 'gaussian' or 'saltpepper'.")
            noisy = np.rint(denoise.add_noise(original, kind, level)).astype(np.uint8)
            save_image(image_id, "noisy", noisy)
            metadata = read_metadata(image_id)
            metadata["noise_kind"] = kind
            temp = metadata_path(image_id).with_suffix(".tmp")
            temp.write_text(json.dumps(metadata), encoding="utf-8")
            temp.replace(metadata_path(image_id))
            unit = "standard deviation" if kind == "gaussian" else "% of pixels"
            return jsonify(noisy_image=as_base64(noisy), message=f"Added {kind} noise ({raw_level:g} {unit}).",
                           metrics=metrics(original, noisy))
    except (ValueError, FileNotFoundError) as exc:
        return error(str(exc), 404 if isinstance(exc, FileNotFoundError) else 400)


@app.post("/api/denoise")
def denoise_image():
    payload = request.get_json(silent=True) or {}
    try:
        with active_image(payload.get("image_id")) as image_id:
            original = read_image(image_id)
            # The current image is the artificial noisy image when present;
            # otherwise denoise the uploaded image directly.
            source = "noisy" if image_path(image_id, "noisy").is_file() else "original"
            input_image = read_image(image_id, source)
            # Salt-and-pepper noise creates isolated black/white pixels. A small
            # median filter removes these outliers before SVD, avoiding the
            # need to discard many singular values and blur the whole image.
            metadata = read_metadata(image_id)
            noise_kind = metadata.get("noise_kind") if source == "noisy" else None
            svd_input = (denoise.median_filter_image(input_image)
                         if noise_kind == "saltpepper" else input_image)
            svd = decompose(svd_input)
            k, reason = denoise.auto_k_denoise_image(svd, svd_input.shape[:2])
            result = denoise.denoise_reconstruct(svd, k)
            if noise_kind == "saltpepper":
                reason = "Median filter removed isolated salt-and-pepper pixels, followed by SVD reconstruction. " + reason
            save_image(image_id, "denoised", result)
            input_values, result_values = metrics(original, input_image), metrics(original, result)
            return jsonify(k_used=k, image=as_base64(result), reason=reason,
                           source=source, input_metrics=input_values, denoised_metrics=result_values,
                           psnr_before=input_values["psnr"], psnr_after=result_values["psnr"],
                           file_sizes=file_size_summary(image_id, "denoised"))
    except (ValueError, FileNotFoundError) as exc:
        return error(str(exc), 404 if isinstance(exc, FileNotFoundError) else 400)


@app.errorhandler(413)
def too_large(_):
    return error(f"Image is too large; maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.", 413)


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    development = os.getenv("FLASK_ENV", "development").lower() == "development"
    app.run(debug=development)

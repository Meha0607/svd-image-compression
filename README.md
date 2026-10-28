# SVD Image Lab

An interactive Flask application for SVD-based image compression and noise
reduction. It accepts PNG, JPG/JPEG, and BMP images up to 10 MB and 1024 pixels
on either side.

## Setup and run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000` in a browser. The UI is served by Flask; no
separate frontend build or development server is needed.

## API and processing notes

- `POST /api/upload` accepts a `file` multipart field and returns an image ID.
- `GET /api/matrix-basics?image_id=<id>` returns the top-left grayscale block's
  rank, nullity, RREF, null-space basis, and Gram--Schmidt basis.
- `POST /api/compress` accepts `image_id`, `mode` (`manual` or `auto`), and
  either integer `k` or an auto method (`energy`, `ratio`, or `elbow`).
- `POST /api/noise` accepts `kind` (`gaussian` or `saltpepper`) and `level`.
  Gaussian level is its standard deviation (0--100); salt-and-pepper level is
  a percentage (0--100) at the API/UI boundary and is converted to the
  module's 0--1 fraction.
- `POST /api/denoise` accepts an image ID and manual or automatic rank choice.
- `POST /api/rank-suggestion` previews an automatic rank without writing an
  image. `GET /api/download/<image_id>/compressed` and `/denoised` download
  the actual processed PNG files.

The backend stores the original, compressed, noisy, and denoised variants as
separate PNG files under `data/<uuid>/`. It records the uploaded file size and
compares it with the processed PNG size; this is deliberately labelled
separately from the theoretical SVD matrix compression ratio. The frontend
uses real API requests (`USE_DUMMY = false`).

Quality cards compare each processed image to the uploaded original: MSE,
PSNR, and channel-averaged local-window SSIM. A PSNR of 100 dB represents an
identical result rather than infinity.

## Tests

```powershell
python -m unittest -v
```

The tests cover grayscale/RGB reconstruction, matrix operations, quality
metric edge cases, invalid/corrupt/oversized uploads, downloads, byte-saving
responses, automatic-rank preview, storage cleanup, and the full upload →
analyse → compress → noise → denoise workflow.

## Configuration

Environment variables are optional; defaults are suitable for local use:

- `FLASK_ENV=development`: set to `production` to disable Flask debug mode.
- `SVD_DATA_DIR=./data`: private writable directory for per-upload data.
- `SVD_MAX_SIDE=1024`: maximum width or height in pixels.
- `SVD_MAX_UPLOAD_BYTES=10485760`: maximum uploaded file size in bytes.
- `SVD_RETENTION_HOURS=24`: age before inactive UUID upload folders are removed; use a negative value to disable cleanup.

Command Prompt example:

```cmd
set FLASK_ENV=production
set SVD_RETENTION_HOURS=48
set SVD_MAX_UPLOAD_BYTES=15728640
waitress-serve --host=127.0.0.1 --port=8000 app:app
```

## Deployment

For a production Windows deployment, use a WSGI server rather than Flask's
debug server:

```powershell
set FLASK_ENV=production
waitress-serve --host=127.0.0.1 --port=8000 app:app
```

Put Waitress behind HTTPS using IIS, Nginx, Caddy, or a managed platform that
terminates TLS and forwards requests to `127.0.0.1:8000`. Do not expose the
private data directory directly. The application removes expired inactive
UUID upload folders during a later upload according to `SVD_RETENTION_HOURS`;
for low-traffic deployments, schedule a small request or restart cycle so
cleanup still runs regularly.

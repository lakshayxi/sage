# Sage demo image for Hugging Face Spaces (Docker SDK, free CPU tier, 16GB
# RAM -- chosen because the cross-encoder reranker (sentence-transformers /
# PyTorch) needs real memory most other free tiers don't offer).
#
# Single image, single URL: FastAPI serves both the API and the built
# frontend static assets, and a pre-ingested demo corpus is baked in
# unconditionally. This repo only ever deploys this one way -- no env-gated
# dual-mode logic, no second deployment path.
#
# Two stages:
#   1. frontend-build (node) -- builds frontend/ (React + Vite) into static
#      assets (frontend/dist).
#   2. runtime (python)      -- installs the Sage package, copies in the
#      built frontend and the pre-ingested demo data, and runs uvicorn.

FROM node:20-slim AS frontend-build
WORKDIR /frontend
# package-lock.json* (glob) tolerates the lockfile not existing yet while
# frontend/ is still under active development elsewhere in this repo; once
# it's committed, `npm ci` uses it for a reproducible install.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
# Vite inlines VITE_* env vars into the built JS at build time, not runtime
# -- there's no way to configure this after the image is built, so it has to
# come in as a build-arg here rather than a Space "Variable" like
# DEMO_ACCESS_KEY (see deploy/huggingface/DEPLOY.md). Empty by default,
# matching DEMO_ACCESS_KEY being unset by default.
ARG VITE_DEMO_ACCESS_KEY=""
ENV VITE_DEMO_ACCESS_KEY=$VITE_DEMO_ACCESS_KEY
RUN npm run build

# "bookworm" pinned explicitly, not plain "slim": chromadb requires
# sqlite3>=3.35, and older Debian bases have shipped sqlite3 older than that,
# which fails at import time with "unsupported version of sqlite3". Mirrors
# the same pin in the sibling reference project's Dockerfile
# (/Users/shay/Desktop/projects/edge/Dockerfile).
FROM python:3.11-slim-bookworm

# HF Spaces containers run as uid 1000 -- create that exact user and set its
# WORKDIR before any COPY, per HF's Docker Spaces permissions guidance
# (huggingface.co/docs/hub/spaces-sdks-docker#permissions), then use
# --chown on every COPY so files land owned by that user instead of root.
RUN useradd -m -u 1000 appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH
WORKDIR $HOME/app

# Application code only -- tests/, docs/, eval-only extras stay out of the
# runtime image.
COPY --chown=appuser pyproject.toml ./
COPY --chown=appuser sage ./sage
COPY --chown=appuser api ./api
COPY --chown=appuser config ./config
COPY --chown=appuser --from=frontend-build /frontend/dist ./frontend/dist

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

# data/{raw,processed,chroma} and db/ are where the app writes/reads at
# runtime (config/settings.py's RAW_DIR/PROCESSED_DIR/CHROMA_DIR/SQLITE_PATH
# -- that module unconditionally mkdir()s all four on import, every process
# start, regardless of whether uploads are enabled). All four are created
# and chowned to appuser here: this RUN still executes as root (USER appuser
# hasn't been set yet), so without the explicit chown, "data" and "db" would
# be root-owned directories that a later --chown=appuser COPY of *files*
# into them doesn't fix -- SQLite's WAL files, Chroma's new segment files,
# and that startup mkdir() all need write access to the containing
# directory itself, not just to a pre-existing file in it, or the app fails
# on its first write / crashes on import at container start.
RUN mkdir -p data/raw data/processed data/chroma db \
    && chown -R appuser:appuser data db

# Pre-ingested demo corpus (Apple/Microsoft/NVIDIA 10-Ks), baked in
# unconditionally so the public demo works with zero setup. Both sources
# always exist as of this Dockerfile (deploy/huggingface/prebuilt/chroma/
# has a .gitkeep placeholder, sage.db is a valid empty SQLite file) even
# before the real ingested data lands, so these COPYs never fail the build --
# see deploy/huggingface/prebuilt/README.md for what replaces the
# placeholders and how.
COPY --chown=appuser deploy/huggingface/prebuilt/chroma/ ./data/chroma/
COPY --chown=appuser deploy/huggingface/prebuilt/sage.db ./db/sage.db

USER appuser

# HF Docker Spaces expect the app to listen on 7860 by default (app_port in
# deploy/huggingface/README.md's frontmatter matches).
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

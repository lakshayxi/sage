# Sage demo image for Hugging Face Spaces (Docker SDK, CPU runtime). The
# cross-encoder reranker needs materially more memory than a static frontend,
# so confirm the selected Space hardware can warm both local models.
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

FROM node:20-slim@sha256:2cf067cfed83d5ea958367df9f966191a942351a2df77d6f0193e162b5febfc0 AS frontend-build
WORKDIR /frontend
# package-lock.json* (glob) tolerates the lockfile not existing yet while
# frontend/ is still under active development elsewhere in this repo; once
# it's committed, `npm ci` uses it for a reproducible install.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# "bookworm" pinned explicitly, not plain "slim": chromadb requires
# sqlite3>=3.35, and older Debian bases have shipped sqlite3 older than that,
# which fails at import time with "unsupported version of sqlite3". Mirrors
# the same pin in the sibling reference project's Dockerfile.
FROM python:3.14-slim-bookworm@sha256:23c59390fc717bf09f9336908199a0ae75d9c4264bf296123f94ad772fea3b52

# HF Spaces containers run as uid 1000 -- create that exact user and set its
# WORKDIR before any COPY, per HF's Docker Spaces permissions guidance
# (huggingface.co/docs/hub/spaces-sdks-docker#permissions), then use
# --chown on every COPY so files land owned by that user instead of root.
RUN useradd -m -u 1000 appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH
WORKDIR $HOME/app

# Dependency metadata is copied before application source so ordinary code
# edits retain the expensive Python/model layers in Docker's build cache.
COPY --chown=appuser pyproject.toml requirements-lock.txt ./

# sentence-transformers pulls in torch as a transitive dependency; resolving
# it from plain PyPI gets the default CUDA-enabled build, which drags in
# several hundred MB of unused nvidia-* wheels on this CPU-only HF Space.
# Preinstalling the pinned CPU wheel from PyTorch's own CPU index first means
# pip already sees torch satisfied when it later resolves Sage's deps, and
# never swaps it for the CUDA build.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.13.0+cpu --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-lock.txt

# Pin and download both local retrieval models while the image is built.
# FastAPI warms them before accepting traffic; without this layer every
# restart depends on the Hub being reachable and makes the first visitor wait
# for two downloads. Runtime is deliberately offline for Hugging Face only --
# Gemini generation still uses normal outbound HTTPS.
ARG SAGE_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
ARG SAGE_EMBEDDING_REVISION="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
ARG SAGE_RERANKER_MODEL="BAAI/bge-reranker-base"
ARG SAGE_RERANKER_REVISION="2cfc18c9415c912f9d8155881c133215df768a70"
ENV SAGE_EMBEDDING_MODEL=$SAGE_EMBEDDING_MODEL \
    SAGE_EMBEDDING_REVISION=$SAGE_EMBEDDING_REVISION \
    SAGE_RERANKER_MODEL=$SAGE_RERANKER_MODEL \
    SAGE_RERANKER_REVISION=$SAGE_RERANKER_REVISION
RUN python -c "import os; from sentence_transformers import CrossEncoder, SentenceTransformer; SentenceTransformer(os.environ['SAGE_EMBEDDING_MODEL'], revision=os.environ['SAGE_EMBEDDING_REVISION']); CrossEncoder(os.environ['SAGE_RERANKER_MODEL'], revision=os.environ['SAGE_RERANKER_REVISION'])" \
    && chown -R appuser:appuser /home/appuser/.cache
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    QUERY_LOGGING_ENABLED=false \
    FORWARDED_ALLOW_IPS="*"

# Application code only -- tests/, docs/, eval-only extras stay out of the
# runtime image. Editable registration is a cheap separate layer after the
# source copy; all dependencies were already resolved from the lock above.
COPY --chown=appuser sage ./sage
COPY --chown=appuser api ./api
COPY --chown=appuser config ./config
COPY --chown=appuser --from=frontend-build /frontend/dist ./frontend/dist
RUN pip install --no-cache-dir -e . --no-deps

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
# always exist as of this Dockerfile and are validated before the deploy
# staging script will upload them. See the prebuilt README for the privacy
# boundary and reproducible rebuild command.
COPY --chown=appuser deploy/huggingface/prebuilt/chroma/ ./data/chroma/
COPY --chown=appuser deploy/huggingface/prebuilt/sage.db ./db/sage.db

USER appuser

# HF Docker Spaces expect the app to listen on 7860 by default (app_port in
# deploy/huggingface/README.md's frontmatter matches).
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:7860/readyz', timeout=3)"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]

# ── Stage 1: builder ──────────────────────────────────────────────────────────
# Installs all Python deps into an isolated venv.
# Build tools and header packages stay in this layer and never reach the final image.
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo-dev zlib1g-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Isolated venv — keeps the final image free of pip internals
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# --- PyTorch CPU-only wheel (~250 MB vs ~800 MB for the default CUDA build) ---
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

# --- Inference-only runtime deps (no mlflow / scikit-learn / matplotlib / pytest) ---
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt


# ── Stage 2: runtime ──────────────────────────────────────────────────────────
# Minimal image — only the venv and app source, nothing else.
FROM python:3.10-slim AS runtime

WORKDIR /app

# Runtime shared libraries for Pillow (no -dev headers needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg62-turbo zlib1g \
    && rm -rf /var/lib/apt/lists/*

# Copy the clean venv from the builder — no compiler, no cache, no build headers
COPY --from=builder /venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy only what the inference server actually needs at runtime
COPY app.py        ./app.py
COPY src/          ./src/
COPY models/       ./models/

ENV MODEL_PATH=models/model.pt
ENV DEVICE=cpu
ENV PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

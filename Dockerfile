FROM python:3.11-slim

WORKDIR /app

# Build tools + Playwright Chromium runtime libs (Debian Trixie t64 names)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    libnss3 \
    libatk1.0-0t64 \
    libatk-bridge2.0-0t64 \
    libcups2t64 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2t64 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Create the HF-required non-root user BEFORE installing browsers and the
# model so everything lands in /home/user/.cache (accessible at runtime).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PLAYWRIGHT_BROWSERS_PATH=/home/user/.cache/ms-playwright

# Install Playwright Chromium as the non-root user so the binary is in
# /home/user/.cache/ms-playwright, where the running process can find it.
RUN python -m playwright install chromium

# Pre-download BAAI/bge-small-en-v1.5 (~130 MB) to avoid cold-start delay.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('BGE model cached.')"

COPY --chown=user:user backend/ .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

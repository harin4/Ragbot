FROM python:3.11-slim

WORKDIR /app

# System deps: gcc/g++ for sentence-transformers; Playwright Chromium runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    libnss3 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libasound2 libpango-1.0-0 libcairo2 libatspi2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Download Playwright's Chromium browser binary
RUN python -m playwright install chromium

# Create the HF-required non-root user BEFORE caching the model so the
# model lands in /home/user/.cache (the path the app sees at runtime).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH

# Pre-download BAAI/bge-small-en-v1.5 (~130 MB) to avoid cold-start delay
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('BGE model cached.')"

COPY --chown=user:user backend/ .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

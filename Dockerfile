FROM python:3.11-slim

WORKDIR /app

# Install playwright Python package first so we can run 'playwright install'
RUN pip install --no-cache-dir playwright==1.49.0

# 'playwright install --with-deps' fails on Debian because it uses Ubuntu font
# package names (ttf-unifont, ttf-ubuntu-font-family) absent on Debian Trixie.
# Install the exact equivalent Debian packages manually instead.
RUN apt-get update && apt-get install -y --no-install-recommends \
    fonts-unifont \
    libglib2.0-0t64 libdbus-1-3 \
    libnspr4 libnss3 \
    libatk1.0-0t64 libatk-bridge2.0-0t64 \
    libcups2t64 libdrm2 libgbm1 \
    libxkbcommon0 libxshmfence1 \
    libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libasound2t64 \
    libpango-1.0-0 libpangocairo-1.0-0 libcairo2 \
    libatspi2.0-0t64 \
    && rm -rf /var/lib/apt/lists/*

# Download Playwright's Chromium binary (system libs installed above).
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN python -m playwright install chromium \
    && chmod -R 755 /ms-playwright

# Install the rest of the application requirements.
COPY backend/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Create the HF-required non-root user (UID 1000).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

COPY --chown=user:user backend/ .

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

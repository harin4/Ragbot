FROM python:3.11-slim

WORKDIR /app

# Install system Chromium — apt handles all required X11/graphics libs
# automatically, and avoids Playwright's CDN binary download which fails
# in HF Space build runners.
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    && rm -rf /var/lib/apt/lists/*

# Prevent playwright from attempting a browser download during pip install.
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1

COPY backend/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Create the HF-required non-root user
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium

COPY --chown=user:user backend/ .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]

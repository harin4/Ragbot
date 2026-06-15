FROM python:3.11-slim

WORKDIR /app

# Install the playwright Python package alone first so we can run
# 'playwright install --with-deps' before the long pip install that
# fills up HF's 50 k-char build log and hides errors.
RUN pip install --no-cache-dir playwright==1.49.0

# Download Playwright's headless Chromium + auto-install its system libs.
# Runs early so its output appears before pip install truncates the log.
# Root-level install into /ms-playwright which is world-readable.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN python -m playwright install --with-deps chromium \
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

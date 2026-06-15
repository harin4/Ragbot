FROM python:3.11-slim

WORKDIR /app

# gcc/g++ required by sentence-transformers and tokenizers
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Pre-download BAAI/bge-small-en-v1.5 (~130 MB) so the container is
# ready immediately — avoids a 40-second cold-start on first request.
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('BGE model cached.')"

COPY . .

EXPOSE 8000

# $PORT is injected by Railway; defaults to 8000 for local docker-compose
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]

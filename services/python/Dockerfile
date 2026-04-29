FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e "."

COPY src/ src/
COPY scripts/ scripts/
COPY models/ models/

CMD ["python", "-m", "src.main"]

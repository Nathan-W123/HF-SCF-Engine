FROM python:3.11-slim

# Build deps for numpy/scipy/numba C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source (includes basis_cache.json so BSE fetch is not needed at runtime)
COPY backend/ .

# Persistent volume for SQLite database will be mounted at /data
RUN mkdir -p /data

ENV PYTHONUNBUFFERED=1
ENV DATABASE_URL=sqlite:////data/hf_scf.db

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

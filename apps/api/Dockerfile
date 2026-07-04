FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8000
ENV ALITIGATOR_RAG_BACKEND=mysql
ENV ALITIGATOR_RAG_CROSS_ENCODER_ENABLED=false

COPY requirements.txt ./
RUN grep -v '^sentence-transformers==' requirements.txt > requirements.runtime.txt \
    && pip install --no-cache-dir -r requirements.runtime.txt

COPY app ./app
COPY scripts ./scripts
COPY sql ./sql

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

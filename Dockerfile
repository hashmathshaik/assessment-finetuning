FROM python:3.11-slim

RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY deployment/ ./deployment/

USER app
ENV MODEL_DIR=/models/bert-seed42 \
    TORCH_THREADS=2 \
    PYTHONUNBUFFERED=1

EXPOSE 8000
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]

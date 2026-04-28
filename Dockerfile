FROM python:3.10-slim

ARG APP_VERSION=dev

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    APP_VERSION=${APP_VERSION} \
    UVICORN_HOST=0.0.0.0 \
    UVICORN_PORT=8000

LABEL org.opencontainers.image.version="${APP_VERSION}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

RUN mkdir -p /app/uploads /app/results

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=25s --retries=3 \
  CMD python -c "import os, sys, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"UVICORN_PORT\", \"8000\")}/api/ready', timeout=3); sys.exit(0)"

CMD ["sh", "-c", "uvicorn scripts.api_server:app --host ${UVICORN_HOST} --port ${UVICORN_PORT}"]

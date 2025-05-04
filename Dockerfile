FROM python:3.11-slim


RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md /app/
RUN pip install --no-cache-dir --upgrade pip poetry==1.8.1 \
    && poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

COPY src /app/src
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["digit-classification"]
CMD ["--help"]

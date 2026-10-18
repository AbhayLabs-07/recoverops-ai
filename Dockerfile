FROM python:3.11.2-slim-bullseye

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN python -m pip install --no-cache-dir --requirement requirements.txt

COPY app ./app
COPY tests ./tests
COPY Makefile README.md .python-version .env.example ./

CMD ["python", "-m", "app.main"]

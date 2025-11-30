FROM python:3.10-slim

WORKDIR /app

# system deps (for plotting/pdf)
RUN apt-get update && apt-get install -y build-essential libcairo2-dev libpango1.0-dev libffi-dev libsndfile1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV PORT=8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]


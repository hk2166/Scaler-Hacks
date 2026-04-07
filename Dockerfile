FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download TextBlob corpora for NLP grading
RUN python -m textblob.download_corpora

COPY . .

RUN mkdir -p environment/graders && \
    touch environment/__init__.py && \
    touch environment/graders/__init__.py

EXPOSE 7860

CMD ["python", "-m", "server.app"]

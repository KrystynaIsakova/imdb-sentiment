# IMDB Sentiment Analysis

NLP project that classifies movie review sentiment (positive/negative) using two approaches: a classical ML pipeline with scikit-learn, and zero-shot classification via the OpenAI API.

## Project Structure

```
imdb_sentiment/
├── data/
│   └── IMDB Dataset.csv       # 50k IMDB reviews (downloaded via kagglehub)
├── models/
│   ├── sentiment_pipeline.joblib
│   └── label_encoder.joblib
├── src/
│   ├── get_data.py            # Download dataset from Kaggle
│   ├── preprocessing.py       # Text cleaning + NLTK lemmatization
│   ├── train.py               # TF-IDF + Logistic Regression pipeline
│   └── api.py                 # FastAPI inference server
└── SentimentOpenAI.ipynb      # GPT-4o-mini zero-shot baseline (90% accuracy on sample)
```

## Approaches

### 1. Classical ML Pipeline
- Text preprocessing: lowercasing, HTML tag removal, stopword filtering, POS-aware lemmatization (NLTK)
- Features: TF-IDF (top 5,000 terms)
- Classifier: Logistic Regression
- Served via FastAPI

### 2. OpenAI Zero-Shot (`SentimentOpenAI.ipynb`)
- Prompts `gpt-4o-mini` to classify each review as `positive` or `negative`
- Achieved **90% accuracy** on a 10-sample test

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the OpenAI notebook, create a `.env` file:

```
OPENAI_API_KEY=your_key_here
```

## Usage

### Download Data

```bash
python src/get_data.py
```

### Train the Model

```bash
python src/train.py
```

Saves `models/sentiment_pipeline.joblib` and `models/label_encoder.joblib`.

### Run the API

```bash
uvicorn src.api:app --reload
```

**Endpoints:**

- `GET /` — health check
- `POST /predict` — classify a review

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This movie was absolutely fantastic!"}'
```

```json
{"sentiment": "positive", "probability": 0.93}
```

## Dataset

[IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) — downloaded automatically via `kagglehub`.

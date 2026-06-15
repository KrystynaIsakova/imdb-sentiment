import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.preprocessing import preprocess_text


DATA_PATH = Path("data/IMDB Dataset.csv")
MODEL_DIR = Path("models")
# MODEL_DIR.mkdir(exist_ok=True)


def main():
    df = pd.read_csv(DATA_PATH)

    X = df["review"]
    y = df["sentiment"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, preprocessor=preprocess_text)),
        ("model", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X, y)

    joblib.dump(pipeline, MODEL_DIR / "sentiment_pipeline.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")

    print("Model saved to models/")


if __name__ == "__main__":
    main()
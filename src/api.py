from fastapi import FastAPI
import joblib

pipeline = joblib.load("models/sentiment_pipeline.joblib")

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Sentiment API is running"}


@app.post("/predict")
def predict(data: dict):
    review = data["review"]

    prediction = pipeline.predict([review])[0]
    probability = max(pipeline.predict_proba([review])[0])

    return {
        "sentiment": prediction,
        "probability": float(probability)
    }
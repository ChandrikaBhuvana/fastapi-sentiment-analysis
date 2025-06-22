from fastapi import FastAPI
from app.schemas import ReviewRequest, PredictionResponse
from app.utils import predict_sentiment

app = FastAPI(title="IMDb Sentiment Classifier API")

@app.get("/")
def root():
    return {"message": "Welcome to the IMDb Sentiment Analysis API!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: ReviewRequest):
    sentiment, confidence = predict_sentiment(request.review)
    return PredictionResponse(sentiment=sentiment, confidence=confidence)

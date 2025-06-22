import re
import joblib

# Load model and vectorizer
model = joblib.load("ml/model.pkl")
vectorizer = joblib.load("ml/vectorizer.pkl")

def clean_text(text):
    text = re.sub(r"<.*?>"," ", text)
    text = re.sub(r"[^a-zA-Z']", " ", text)
    text = text.lower()
    return text

def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0][prediction]
    sentiment = "positive" if prediction == 1 else "negative"
    return sentiment, float(proba)
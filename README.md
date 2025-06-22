# 🎬 IMDB Movie Review Sentiment Analysis API

This project uses machine learning to classify IMDB movie reviews as **positive** or **negative**. The model is served through a **FastAPI** application with interactive Swagger documentation.

---

## 🚀 Features

- ✅ Logistic Regression classifier trained on cleaned IMDB reviews  
- ✅ Text preprocessing with TF-IDF vectorization  
- ✅ FastAPI-based REST API with request validation (Pydantic)  
- ✅ Interactive Swagger UI for testing  
- ✅ Clean and modular project structure  
- ✅ Local deployment with Uvicorn

---

## 🧠 Model Info

- **Algorithm:** Logistic Regression  
- **Accuracy:** 89.35% on test set  
- **Vectorizer:** TF-IDF (bigrams, stop words removed, max features = 10,000)  
- **Data Source:** IMDB Movie Review Dataset

---


## ⚙️ Setup Instructions

1. **Clone the repository**
2. **Create and activate a virtual environment**
3. **Install dependencies**

    pip install -r requirements.txt

4. **Start the FastAPI server**

    uvicorn app.main:app --reload

5. **Visit Swagger UI**

    http://127.0.0.1:8000/docs

---

## 📬 API Usage

### POST /predict

#### Request Body

    {
      "review": "The movie was absolutely amazing!"
    }

#### Response

    {
      "sentiment": "positive"
    }

---

## 📌 Notes

- This is a local deployment project meant to understand the basics of FastAPI and ML integration.
- Future scope includes Dockerizing the app and deploying it on Render or Railway.

---

## 👩‍💻 Author

Bhuvana Chandrika



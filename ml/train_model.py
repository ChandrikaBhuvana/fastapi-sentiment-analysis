import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

#Loading dataset
df = pd.read_csv('E:/fastapi_sentiment_analysis/data/imdb.csv')

#Preview
# print("Shape of dataset: ", df.shape)
# print("\nFirst few rows: ")
# print(df.head())

#Cleaning the text
def clean_text(text):
    text = re.sub(r"<.*?>"," ", text) #Remove HTML tags
    text = re.sub(r"[^a-zA-Z']", " ", text) # Keep letters and apostrophes
    text = text.lower() #Lowercase
    return text

df['review'] = df['review'].apply(clean_text)

#Preview2
# print("Shape of dataset: ", df.shape)
# print("\nFirst few rows: ")
# print(df.head())

#Convert Sentiment to numeric
df['sentiment'] = df['sentiment'].map({'positive' : 1, 'negative' : 0})


#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment'], test_size=0.2, random_state=42)

#Preview3
# print("Shape of dataset: ", X_train.shape)
# print("Shape of dataset: ", X_test.shape)
# print("Shape of dataset: ", y_train.shape)
# print("Shape of dataset: ", y_test.shape)
# print("\nFirst few rows: ")
# print(X_train.head())
# print(X_test.head())
# print(y_train.head())
# print(y_test.head())

#Vectorizing the text
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# #Preview3
# print("Shape of dataset: ", X_train_vec.shape)
# print("Shape of dataset: ", X_test_vec.shape)
# print("Shape of dataset: ", y_train.shape)
# print("Shape of dataset: ", y_test.shape)

# Model dictionary
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Multinomial Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC()
}

best_model_name = ""
best_model_score = 0.0
best_model = None

# Train & evaluate
for name, model in models.items():
    print(f"\n🔍 Training: {name}")
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy ({name}): {acc:.4f}")
    print("📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    if acc > best_model_score:
        best_model_score = acc
        best_model_name = name
        best_model = model

# Save best model and vectorizer
if best_model_name == "Logistic Regression":
    joblib.dump(best_model, "ml/model.pkl")
    joblib.dump(vectorizer, "ml/vectorizer.pkl")
    print(f"\n💾 Saved best model: {best_model_name} (Accuracy: {best_model_score:.4f}) to 'ml/model.pkl'")
    print("💾 Saved vectorizer to 'ml/vectorizer.pkl'")
else:
    print(f"\n⚠️ Best model was {best_model_name} but only Logistic Regression is saved for deployment.")
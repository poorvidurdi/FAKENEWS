import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# =================================================
# STEP 0: Resolve Project Paths
# =================================================

# Project root: FAKENEWS/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models", "text")

os.makedirs(MODEL_DIR, exist_ok=True)


# =================================================
# STEP 1: Load Already-Split Dataset
# =================================================

#train_df = pd.read_csv(os.path.join(DATA_DIR, "train_text.csv"))
#test_df = pd.read_csv(os.path.join(DATA_DIR, "test_text.csv"))

#print("Train shape:", train_df.shape)
#print("Test shape:", test_df.shape)
# =================================================
# STEP 1: Load Custom Text Dataset
# =================================================

df = pd.read_csv(os.path.join(DATA_DIR, "new_text.csv"))

print("Total dataset size:", df.shape)
print(df["label"].value_counts())
# =================================================
# STEP 2: Train-Test Split
# =================================================

X = df["title"] + " " + df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=400,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])


# =================================================
# STEP 2: Column Selection & Missing Values
# =================================================

# Drop unused column
#X_train_text = X_train.drop(columns=["author"], errors="ignore")
#test_df = test_df.drop(columns=["author"], errors="ignore")

# Fill missing text
#X_train_text["title"] = X_train["title"].fillna("")
#train_df["text"] = train_df["text"].fillna("")

#test_df["title"] = test_df["title"].fillna("")
#test_df["text"] = test_df["text"].fillna("")

# Combine title + text
#train_df["content"] = train_df["title"] + " " + train_df["text"]
#test_df["content"] = test_df["title"] + " " + test_df["text"]

#X_train_text = train_df["content"]
#y_train = train_df["label"]

#X_test_text = test_df["content"]


# =================================================
# STEP 3: Feature Extraction (TF-IDF)
# =================================================

tfidf = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    max_features=5000,
    ngram_range=(1, 1)
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# =================================================
# STEP 4: Train Text Classification Model
# =================================================

#model = LogisticRegression(max_iter=1000)
model = LogisticRegression(
    max_iter=1000,
    C=0.5,          # stronger regularization
    solver="liblinear",
    class_weight="balanced"
)

model.fit(X_train_tfidf, y_train)


# =================================================
# STEP 5: Training Evaluation
# =================================================

test_preds = model.predict(X_test_tfidf)

print("\nTest Accuracy:", accuracy_score(y_test, test_preds))
print("\nClassification Report (Test):")
print(classification_report(y_test, test_preds))


# =================================================
# STEP 6: Predict on Test Set
# =================================================

test_predictions = model.predict(X_test_tfidf)
test_probabilities = model.predict_proba(X_test_tfidf)[:, 1]

results_df = pd.DataFrame({
    "text": X_test,
    "true_label": y_test,
    "predicted_label": test_predictions,
    "fake_probability": test_probabilities
})


# =================================================
# STEP 7: Save Outputs for Later Stages
# =================================================

# Save predictions
results_df.to_csv(
    os.path.join(BASE_DIR, "results", "sample_outputs", "custom_text_predictions.csv"),
    index=False
)

# Save model and vectorizer
with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

with open(os.path.join(MODEL_DIR, "text_classifier.pkl"), "wb") as f:
    pickle.dump(model, f)

print("\nText fake news detection pipeline completed successfully.")
# =================================================
# STEP 8: Text Explainability
# =================================================

feature_names = tfidf.get_feature_names_out()
coefficients = model.coef_[0]

# Create dataframe of words and their weights
importance_df = pd.DataFrame({
    "word": feature_names,
    "coefficient": coefficients
})

# Top words pushing towards FAKE
top_fake_words = importance_df.sort_values(
    by="coefficient", ascending=False
).head(20)

# Top words pushing towards REAL
top_real_words = importance_df.sort_values(
    by="coefficient", ascending=True
).head(20)

print("\nTop words indicating FAKE news:")
print(top_fake_words)

print("\nTop words indicating REAL news:")
print(top_real_words)

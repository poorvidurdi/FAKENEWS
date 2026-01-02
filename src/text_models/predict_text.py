import os
import pickle

# ---------------------------------------------
# Resolve paths
# ---------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models", "text")

# ---------------------------------------------
# Load saved vectorizer and model
# ---------------------------------------------

with open(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    tfidf = pickle.load(f)

with open(os.path.join(MODEL_DIR, "text_classifier.pkl"), "rb") as f:
    model = pickle.load(f)

# ---------------------------------------------
# User input
# ---------------------------------------------

print("\nEnter news text (press Enter twice to submit):")

lines = []
while True:
    line = input()
    if line == "":
        break
    lines.append(line)

user_text = " ".join(lines)

if not user_text.strip():
    print("No text entered.")
    exit()

# ---------------------------------------------
# Prediction
# ---------------------------------------------

text_vector = tfidf.transform([user_text])
prediction = model.predict(text_vector)[0]
probability = model.predict_proba(text_vector)[0][1]

# ---------------------------------------------
# Explainability: Suspicious words
# ---------------------------------------------

feature_names = tfidf.get_feature_names_out()

# Get model coefficients
coefficients = model.coef_[0]

# Transform user text
text_vector = tfidf.transform([user_text])

# Get non-zero feature indices (words present)
non_zero_indices = text_vector.nonzero()[1]

word_contributions = []

for idx in non_zero_indices:
    word = feature_names[idx]
    contribution = text_vector[0, idx] * coefficients[idx]
    word_contributions.append((word, contribution))

# Sort by contribution toward FAKE (positive side)
word_contributions = sorted(
    word_contributions,
    key=lambda x: x[1],
    reverse=True
)

# Select top suspicious words
suspicious_words = [w for w, c in word_contributions if c > 0][:5]
if probability < 0.40:
    print("\n🟩 Prediction: REAL NEWS")

elif probability < 0.60:
    print("\n🟨 Prediction: UNCERTAIN (Needs Verification)")
    if suspicious_words:
        print("⚠️ Potentially misleading terms:", ", ".join(suspicious_words))

else:
    print("\n🟥 Prediction: FAKE NEWS")
    if suspicious_words:
        print("🚩 Suspicious terms influencing prediction:", ", ".join(suspicious_words))

print(f"Confidence (Fake Probability): {probability:.2f}")

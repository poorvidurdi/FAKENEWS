import os
import re
import pandas as pd
import requests
from tqdm import tqdm
from PIL import Image
from io import BytesIO

# -----------------------
# Paths
# -----------------------
RAW_CSV = "data/processed/multimodal.csv"
OUTPUT_CSV = "data/processed/multimodal_preprocessed.csv"
IMAGE_DIR = "data/images"

os.makedirs(IMAGE_DIR, exist_ok=True)

# -----------------------
# Text cleaning function
# -----------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)          # remove HTML
    text = re.sub(r"http\S+", " ", text)        # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)       # keep alphabets only
    text = re.sub(r"\s+", " ", text).strip()
    return text

# -----------------------
# Image download function
# -----------------------
def download_image(url, image_path):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img.save(image_path)
        return True
    except:
        return False

# -----------------------
# Main preprocessing
# -----------------------
def main():
    df = pd.read_csv(RAW_CSV)

    # Drop duplicates
    df.drop_duplicates(subset=["title", "text"], inplace=True)

    # Drop missing values
    df.dropna(subset=["text", "top_img", "label"], inplace=True)

    # Combine title and text
    df["combined_text"] = df["title"] + " " + df["text"]
    df["combined_text"] = df["combined_text"].apply(clean_text)

    image_paths = []

    print("Downloading images...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = os.path.join(IMAGE_DIR, f"{idx}.jpg")
        success = download_image(row["top_img"], image_path)
        image_paths.append(image_path if success else None)

    df["image_path"] = image_paths

    # Drop rows where image download failed
    df.dropna(subset=["image_path"], inplace=True)

    # Keep final columns only
    df = df[["combined_text", "image_path", "label"]]

    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Preprocessing complete. Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

import os
import pandas as pd

OUTPUT_PATH = "data/processed/multimodal.csv"

def load_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Keep only required columns
    df = df[["title", "text", "top_img"]]

    # Assign label from filename
    filename = os.path.basename(csv_path).lower()
    if "fake" in filename:
        df["label"] = 1
    elif "real" in filename:
        df["label"] = 0
    else:
        raise ValueError(f"Cannot determine label from filename: {filename}")

    print(f"Loaded {csv_path} | Label = {df['label'].iloc[0]}")
    return df

def main():
    all_dfs = []

    # PolitiFact files
    all_dfs.append(load_csv("data/PolitiFact/PolitiFact_fake_news_content.csv"))
    all_dfs.append(load_csv("data/PolitiFact/PolitiFact_real_news_content.csv"))

    # BuzzFeed files
    all_dfs.append(load_csv("data/BuzzFeed/Buzzfeed_fake_news_content.csv"))
    all_dfs.append(load_csv("data/BuzzFeed/Buzzfeed_real_news_content.csv"))

    # Merge all
    final_df = pd.concat(all_dfs, axis=0)

    # Drop unusable rows
    final_df.dropna(subset=["text", "top_img"], inplace=True)

    # Shuffle
    final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    os.makedirs("data/processed", exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)

    print("\nMerged dataset saved to:", OUTPUT_PATH)
    print("Total samples:", len(final_df))

if __name__ == "__main__":
    main()

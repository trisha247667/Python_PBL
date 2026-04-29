import pandas as pd
import string
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

def main():
    df = pd.read_csv("spam_dataset_1000.csv")
    df["cleaned_message"] = df["message"].apply(clean_text)

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["cleaned_message"])
    y = df["label"]

    pd.DataFrame(X.toarray()).to_csv("features.csv", index=False)
    y.to_csv("labels.csv", index=False)

    print("Phase 2 complete")

if __name__ == "__main__":
    main()

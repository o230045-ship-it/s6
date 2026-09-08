import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def train_congestion_model():
    data_path = "../data/railway_data.csv"
    df = pd.read_csv(data_path)
    
    feature_cols = [
        "train_count",
        "track_utilization",
        "block_duration",
        "time"
    ]
    X = df[feature_cols]
    y = df["congestion"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    classifier.fit(X_train, y_train)
    
    preds = classifier.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"=== Congestion Classifier Accuracy: {acc * 100:.2f}% ===")
    print(classification_report(y_test, preds))
    
    os.makedirs("../models", exist_ok=True)
    joblib.dump(classifier, "../models/congestion_model.pkl")
    print("Saved congestion model to ../models/congestion_model.pkl")

if __name__ == "__main__":
    train_congestion_model()

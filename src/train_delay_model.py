import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_delay_model():
    data_path = "../data/railway_data.csv"
    df = pd.read_csv(data_path)
    
    feature_cols = [
        "train_count",
        "track_utilization",
        "block_duration",
        "time",
        "train_priority",
        "previous_delay"
    ]
    X = df[feature_cols]
    y = df["expected_delay"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    
    print("=== Delay Regressor Performance ===")
    print(f"MAE:  {mae:.2f} mins")
    print(f"RMSE: {rmse:.2f} mins")
    print(f"R²:   {r2:.4f}")
    
    os.makedirs("../models", exist_ok=True)
    joblib.dump(model, "../models/delay_model.pkl")
    print("Saved delay model to ../models/delay_model.pkl")

if __name__ == "__main__":
    train_delay_model()

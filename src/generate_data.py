import os
import numpy as np
import pandas as pd

def generate_synthetic_railway_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # 1. Feature distributions
    time_of_day = np.random.randint(0, 24, size=n_samples) # Hour of day (0-23)
    train_priority = np.random.choice([1, 2, 3], size=n_samples, p=[0.5, 0.35, 0.15]) # 1: Goods/Local, 2: Express, 3: Vande Bharat/Rajdhani
    block_duration = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.3, 0.35, 0.2, 0.1, 0.05]) # Hours blocked
    
    # Peak hour effect on traffic (8-11 AM and 17-21 PM)
    is_peak = ((time_of_day >= 8) & (time_of_day <= 11)) | ((time_of_day >= 17) & (time_of_day <= 21))
    train_count = np.where(is_peak, np.random.randint(12, 26, size=n_samples), np.random.randint(3, 15, size=n_samples))
    
    # Track utilization (%) correlated with train count
    base_utilization = (train_count / 25.0) * 85.0
    utilization = np.clip(base_utilization + np.random.normal(0, 6, size=n_samples), 15, 99).round(1)
    
    # Cascading delay from previous section
    previous_delay = np.clip(np.random.exponential(scale=8, size=n_samples), 0, 90).round(1)
    
    # 2. Delay Formula with realistic noise
    noise = np.random.normal(0, 4.0, size=n_samples)
    expected_delay = (
        train_count * 1.45 +
        utilization * 0.22 +
        block_duration * 5.2 +
        previous_delay * 0.45 +
        train_priority * 2.8 +
        (is_peak.astype(int) * 6.0) +
        noise
    )
    expected_delay = np.clip(expected_delay, 2.0, 180.0).round(2)
    
    # 3. Congestion Classification Rules
    # Multi-factor score combining utilization and train volume
    congestion_score = (utilization * 0.65) + ((train_count / 25.0) * 100 * 0.35)
    congestion_labels = []
    for score in congestion_score:
        if score < 50:
            congestion_labels.append("LOW")
        elif score <= 75:
            congestion_labels.append("MEDIUM")
        else:
            congestion_labels.append("HIGH")
            
    df = pd.DataFrame({
        "train_count": train_count,
        "track_utilization": utilization,
        "block_duration": block_duration,
        "time": time_of_day,
        "train_priority": train_priority,
        "previous_delay": previous_delay,
        "congestion": congestion_labels,
        "expected_delay": expected_delay
    })
    
    return df

if __name__ == "__main__":
    os.makedirs("../data", exist_ok=True)
    df = generate_synthetic_railway_data(12000)
    output_path = "../data/railway_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records saved to {output_path}")
    print(df.head())

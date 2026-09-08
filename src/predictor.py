import os
import joblib
import pandas as pd

# Resolve paths relative to script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DELAY_MODEL_PATH = os.path.join(BASE_DIR, "../models/delay_model.pkl")
CONGESTION_MODEL_PATH = os.path.join(BASE_DIR, "../models/congestion_model.pkl")

# Lazy-loaded singletons
_delay_model = None
_congestion_model = None

def get_models():
    global _delay_model, _congestion_model
    if _delay_model is None:
        _delay_model = joblib.load(DELAY_MODEL_PATH)
    if _congestion_model is None:
        _congestion_model = joblib.load(CONGESTION_MODEL_PATH)
    return _delay_model, _congestion_model

def predict_delay(
    train_count: int,
    track_utilization: float,
    block_duration: float,
    time: int,
    train_priority: int = 2,
    previous_delay: float = 0.0
) -> float:
    """Predicts estimated delay in minutes for a given section block."""
    delay_model, _ = get_models()
    input_df = pd.DataFrame([{
        "train_count": train_count,
        "track_utilization": track_utilization,
        "block_duration": block_duration,
        "time": time,
        "train_priority": train_priority,
        "previous_delay": previous_delay
    }])
    pred = delay_model.predict(input_df)[0]
    return float(round(pred, 2))

def predict_congestion(
    train_count: int,
    track_utilization: float,
    block_duration: float,
    time: int
) -> str:
    """Predicts congestion state: 'LOW', 'MEDIUM', or 'HIGH'."""
    _, congestion_model = get_models()
    input_df = pd.DataFrame([{
        "train_count": train_count,
        "track_utilization": track_utilization,
        "block_duration": block_duration,
        "time": time
    }])
    return str(congestion_model.predict(input_df)[0])

def calculate_impact(
    delay: float,
    priority: int,
    congestion: str,
    diversion_penalty: float = 0.0
) -> float:
    """
    Computes unified impact score.
    Lower score indicates a better mitigation or maintenance window.
    """
    priority_weight = {1: 1.0, 2: 1.5, 3: 2.0}
    congestion_penalty = {"LOW": 5.0, "MEDIUM": 15.0, "HIGH": 30.0}
    
    w_p = priority_weight.get(priority, 1.0)
    c_p = congestion_penalty.get(congestion.upper(), 15.0)
    
    score = (delay * w_p) + c_p + diversion_penalty
    return float(round(score, 2))

def evaluate_maintenance_block(
    train_count: int,
    track_utilization: float,
    block_duration: float,
    time: int,
    train_priority: int = 2,
    previous_delay: float = 0.0,
    diversion_penalty: float = 0.0
) -> dict:
    """Comprehensive analysis dictionary for Member 3."""
    delay = predict_delay(
        train_count, track_utilization, block_duration, time, train_priority, previous_delay
    )
    congestion = predict_congestion(
        train_count, track_utilization, block_duration, time
    )
    impact = calculate_impact(delay, train_priority, congestion, diversion_penalty)
    
    return {
        "predicted_delay_mins": delay,
        "congestion_level": congestion,
        "impact_score": impact,
        "recommendation": "APPROVED" if impact < 65 else ("CAUTION" if impact < 110 else "REJECTED")
    }

if __name__ == "__main__":
    result = evaluate_maintenance_block(
        train_count=15,
        track_utilization=85.0,
        block_duration=2.0,
        time=9,
        train_priority=3,
        previous_delay=10.0,
        diversion_penalty=20.0
    )
    print("Sample Evaluation output for Member 3:")
    print(result)

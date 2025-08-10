import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
import joblib

def train(features_df, target_col="score_avg"):
    X = features_df.drop(columns=[target_col])
    y = features_df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"Test RMSE: {rmse:.4f}")

    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "xgb_grade_predictor.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    return model

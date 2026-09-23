import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def nash_sutcliffe_efficiency(y_true, y_pred):
    """
    Computes Nash-Sutcliffe Efficiency (NSE) coefficient used in hydrology.
    NSE = 1 - sum((y_true - y_pred)^2) / sum((y_true - y_mean)^2)
    """
    y_mean = np.mean(y_true)
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum((y_true - y_mean) ** 2)
    return 1.0 - (num / den)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(BASE_DIR, "data", "khadakwasla_evaporation_dataset.csv")
DEFAULT_OUTPUT_DIR = os.path.join(BASE_DIR, "models")

def train_and_evaluate_models(csv_path=DEFAULT_CSV, output_dir=DEFAULT_OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Feature Engineering for ML models
    df['doy'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    
    # Feature matrix X and target y
    features = [
        'temp_max_c', 'temp_min_c', 'temp_mean_c', 
        'humidity_pct', 'wind_speed_2m_ms', 'solar_radiation_mj', 
        'sunshine_hours', 'pressure_hpa', 'vpd_kpa', 'net_radiation_mj', 'doy'
    ]
    
    X = df[features]
    y = df['evaporation_pan_mm']
    
    # Time-based split (80% train, 20% test) to prevent temporal data leakage
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Physics Model Benchmark (Penman Monteith calibrated)
    # Calibrated physics benchmark: E_pan_pred = Penman_E0 / 0.75
    penman_pred_test = (df['penman_evaporation_mm'].iloc[split_idx:] / 0.75).values
    
    models = {
        "FAO-56 Penman Monteith (Physics Baseline)": None, # Evaluated separately
        "Ridge Regression": Pipeline([('scaler', StandardScaler()), ('model', Ridge(alpha=1.0))]),
        "Support Vector Regressor (SVR)": Pipeline([('scaler', StandardScaler()), ('model', SVR(C=10.0, epsilon=0.1))]),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42),
        "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=150, max_depth=14, random_state=42),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=5, random_state=42),
        "Multi-Layer Perceptron (ANN)": Pipeline([('scaler', StandardScaler()), ('model', MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42))])
    }
    
    results = []
    trained_model_dict = {}
    test_predictions = {'Actual_Pan_Evaporation': y_test.values, 'date': df['date'].iloc[split_idx:].values}
    
    # 1. Evaluate Physics Baseline first
    r2_phys = r2_score(y_test, penman_pred_test)
    rmse_phys = np.sqrt(mean_squared_error(y_test, penman_pred_test))
    mae_phys = mean_absolute_error(y_test, penman_pred_test)
    nse_phys = nash_sutcliffe_efficiency(y_test.values, penman_pred_test)
    mape_phys = np.mean(np.abs((y_test.values - penman_pred_test) / y_test.values)) * 100
    
    results.append({
        "Model": "FAO-56 Penman Monteith (Physics Baseline)",
        "R2 Score": round(r2_phys, 4),
        "RMSE (mm/day)": round(rmse_phys, 4),
        "MAE (mm/day)": round(mae_phys, 4),
        "NSE": round(nse_phys, 4),
        "MAPE (%)": round(mape_phys, 2)
    })
    test_predictions["FAO-56 Penman Monteith (Physics Baseline)"] = penman_pred_test
    
    # 2. Train and Evaluate ML/DL Models
    for name, model in models.items():
        if model is None:
            continue
            
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        nse = nash_sutcliffe_efficiency(y_test.values, preds)
        mape = np.mean(np.abs((y_test.values - preds) / y_test.values)) * 100
        
        results.append({
            "Model": name,
            "R2 Score": round(r2, 4),
            "RMSE (mm/day)": round(rmse, 4),
            "MAE (mm/day)": round(mae, 4),
            "NSE": round(nse, 4),
            "MAPE (%)": round(mape, 2)
        })
        
        trained_model_dict[name] = model
        test_predictions[name] = preds
        
    leaderboard_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)
    
    print("\n=== MODEL PERFORMANCE LEADERBOARD ===")
    print(leaderboard_df.to_string(index=False))
    
    # Save Leaderboard & Models
    leaderboard_df.to_csv(os.path.join(output_dir, "model_performance_leaderboard.csv"), index=False)
    
    # Save trained ML artifacts
    artifacts = {
        'models': trained_model_dict,
        'features': features,
        'leaderboard': leaderboard_df,
        'test_predictions': pd.DataFrame(test_predictions)
    }
    joblib.dump(artifacts, os.path.join(output_dir, "trained_models.pkl"))
    print(f"\nSaved trained models and evaluation metrics to: {output_dir}")
    return leaderboard_df

if __name__ == "__main__":
    train_and_evaluate_models()

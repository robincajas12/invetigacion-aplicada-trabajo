import os
import pandas as pd
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import torch
import torch.nn as nn
import torch.optim as optim

# Prevent multithreading overload on 8GB RAM CPU
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
torch.set_num_threads(1)

# Set plotting style
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'

def load_and_preprocess_data():
    print("Loading and preparing Clash Royale dataset...")
    data_dir = "data"
    battles_path = os.path.join(data_dir, "battlesStaging_12282020_WL_tagged.csv")
    wincons_df = pd.read_csv(os.path.join(data_dir, "Wincons.csv"))
    
    # Define archetypes based on wincons
    beatdown_wincons = {"Golem", "Lava Hound", "Giant", "Goblin Giant", "Electro Giant", "Elixir Golem"}
    cycle_wincons = {"Hog Rider", "Goblin Barrel", "Miner", "Wall Breakers", "Skeleton Barrel"}
    siege_wincons = {"X-Bow", "Mortar"}
    bridge_spam_wincons = {"Battle Ram", "Ram Rider", "Royal Giant"}
    control_wincons = {"Graveyard", "Three Musketeers"}
    
    archetypes = ["Beatdown", "Cycle/Control Rápido", "Asedio (Siege)", "Bridge Spam", "Control Lento", "Otros/Híbridos"]
    
    wincon_id_to_archetype = {}
    for _, row in wincons_df.iterrows():
        cid = int(row['card_id'])
        name = row['card_name']
        if name in beatdown_wincons:
            wincon_id_to_archetype[cid] = "Beatdown"
        elif name in cycle_wincons:
            wincon_id_to_archetype[cid] = "Cycle/Control Rápido"
        elif name in siege_wincons:
            wincon_id_to_archetype[cid] = "Asedio (Siege)"
        elif name in bridge_spam_wincons:
            wincon_id_to_archetype[cid] = "Bridge Spam"
        elif name in control_wincons:
            wincon_id_to_archetype[cid] = "Control Lento"
            
    # Load 50,000 battles as requested
    max_rows = 50000
    cols = ['battleTime']
    for prefix in ['winner', 'loser']:
        for i in range(1, 9):
            cols.append(f'{prefix}.card{i}.id')
            
    df = pd.read_csv(battles_path, usecols=cols, nrows=max_rows)
    df = df.sort_values(by='battleTime').reset_index(drop=True)
    
    winner_card_cols = [f'winner.card{i}.id' for i in range(1, 9)]
    loser_card_cols = [f'loser.card{i}.id' for i in range(1, 9)]
    
    winner_ids = df[winner_card_cols].fillna(0).astype(int).values
    loser_ids = df[loser_card_cols].fillna(0).astype(int).values
    
    winner_archs = []
    loser_archs = []
    
    for i in range(len(df)):
        w_arch = "Otros/Híbridos"
        for cid in winner_ids[i]:
            if cid in wincon_id_to_archetype:
                w_arch = wincon_id_to_archetype[cid]
                break
        winner_archs.append(w_arch)
        
        l_arch = "Otros/Híbridos"
        for cid in loser_ids[i]:
            if cid in wincon_id_to_archetype:
                l_arch = wincon_id_to_archetype[cid]
                break
        loser_archs.append(l_arch)
        
    df['winner_arch'] = winner_archs
    df['loser_arch'] = loser_archs
    
    # Dynamic grouping into steps of 1000 battles to form a 50-step time series
    step_size = 1000
    df['step'] = df.index // step_size
    n_steps = len(df) // step_size
    df = df[df['step'] < n_steps]
    
    w_counts = df.groupby(['step', 'winner_arch']).size().unstack(fill_value=0)
    l_counts = df.groupby(['step', 'loser_arch']).size().unstack(fill_value=0)
    
    w_counts = w_counts.reindex(columns=archetypes, fill_value=0)
    l_counts = l_counts.reindex(columns=archetypes, fill_value=0)
    
    use_counts = w_counts + l_counts
    use_rates = use_counts / (step_size * 2)
    
    # Store battleTime representative per step
    times = df.groupby('step')['battleTime'].first()
    
    df_ts = pd.DataFrame(use_rates, columns=archetypes)
    df_ts['ds'] = pd.to_datetime(times).dt.tz_localize(None)
    
    return df_ts, archetypes

# LSTM Network
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

# GRU Network
class GRUForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(GRUForecaster, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.linear(out[:, -1, :])
        return out

def run_experimentation():
    df_ts, archetypes = load_and_preprocess_data()
    n_features = len(archetypes)
    
    # Save the prepared time series
    df_ts.to_csv("data/metagame_time_series.csv", index=False)
    
    # Prepare inputs for sequential models (window size = 5)
    seq_len = 5
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_ts[archetypes])
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_len):
        X.append(scaled_data[i : i + seq_len])
        y.append(scaled_data[i + seq_len])
    X, y = np.array(X), np.array(y)
    
    # Split chronologically (last 20% for test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    y_test_orig = scaler.inverse_transform(y_test)
    
    results = {}
    
    # ---------------- 1. Random Forest ----------------
    print("Training Random Forest Regressor...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=1)
    rf.fit(X_train_flat, y_train)
    rf_preds = rf.predict(X_test_flat)
    results["Random Forest"] = scaler.inverse_transform(rf_preds)
    
    # ---------------- 2. XGBoost ----------------
    print("Training XGBoost Regressor...")
    xgb_preds = []
    for idx in range(n_features):
        xgb_m = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, n_jobs=1)
        xgb_m.fit(X_train_flat, y_train[:, idx])
        xgb_preds.append(xgb_m.predict(X_test_flat))
    xgb_preds = np.column_stack(xgb_preds)
    results["XGBoost"] = scaler.inverse_transform(xgb_preds)
    
    # ---------------- 3. LightGBM ----------------
    print("Training LightGBM Regressor...")
    lgb_preds = []
    for idx in range(n_features):
        lgb_m = lgb.LGBMRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, random_state=42, n_jobs=1, verbose=-1)
        lgb_m.fit(X_train_flat, y_train[:, idx])
        lgb_preds.append(lgb_m.predict(X_test_flat))
    lgb_preds = np.column_stack(lgb_preds)
    results["LightGBM"] = scaler.inverse_transform(lgb_preds)
    
    # ---------------- 4. LSTM ----------------
    print("Training LSTM network...")
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    
    lstm_model = LSTMForecaster(input_dim=n_features, hidden_dim=48, output_dim=n_features, num_layers=2)
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.008, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(250):
        lstm_model.train()
        optimizer.zero_grad()
        out = lstm_model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        
    lstm_model.eval()
    with torch.no_grad():
        lstm_preds = lstm_model(X_test_t).numpy()
    results["LSTM"] = scaler.inverse_transform(lstm_preds)
    
    # ---------------- 5. GRU ----------------
    print("Training GRU network...")
    gru_model = GRUForecaster(input_dim=n_features, hidden_dim=48, output_dim=n_features, num_layers=2)
    optimizer = optim.Adam(gru_model.parameters(), lr=0.008, weight_decay=1e-4)
    
    for epoch in range(250):
        gru_model.train()
        optimizer.zero_grad()
        out = gru_model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        
    gru_model.eval()
    with torch.no_grad():
        gru_preds = gru_model(X_test_t).numpy()
    results["GRU"] = scaler.inverse_transform(gru_preds)
    
    # ---------------- 6. Prophet ----------------
    print("Training Facebook Prophet...")
    prophet_preds = []
    test_dates = df_ts['ds'].iloc[seq_len + train_size:].reset_index(drop=True)
    
    for idx, arch in enumerate(archetypes):
        train_df = pd.DataFrame({
            'ds': df_ts['ds'].iloc[:seq_len + train_size],
            'y': df_ts[arch].iloc[:seq_len + train_size]
        })
        
        m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
        m.fit(train_df)
        
        future = pd.DataFrame({'ds': test_dates})
        forecast = m.predict(future)
        prophet_preds.append(forecast['yhat'].values)
        
    prophet_preds = np.column_stack(prophet_preds)
    results["Prophet"] = prophet_preds
    
    # ---------------- Calculate Metrics ----------------
    metrics_list = []
    for model_name, preds in results.items():
        mae = mean_absolute_error(y_test_orig, preds)
        rmse = np.sqrt(mean_squared_error(y_test_orig, preds))
        r2 = r2_score(y_test_orig.flatten(), preds.flatten())
        
        metrics_list.append({
            "Modelo": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })
        
    df_metrics = pd.DataFrame(metrics_list)
    print("\n=== COMPARACIÓN DE MODELOS ===")
    print(df_metrics.to_markdown(index=False))
    df_metrics.to_csv("data/all_models_metrics.csv", index=False)
    
    # ---------------- Save Plots ----------------
    cycle_idx = archetypes.index('Cycle/Control Rápido')
    beatdown_idx = archetypes.index('Beatdown')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True)
    
    x_steps = np.arange(len(y_test_orig))
    
    colors = {
        "Random Forest": "#e377c2",
        "XGBoost": "#1f77b4",
        "LightGBM": "#ff7f0e",
        "LSTM": "#2ca02c",
        "GRU": "#9467bd",
        "Prophet": "#d62728"
    }
    
    # Cycle plot
    ax1.plot(x_steps, y_test_orig[:, cycle_idx], label='Real (Histórico)', color='black', linewidth=2.2, marker='o')
    for model_name, preds in results.items():
        ax1.plot(x_steps, preds[:, cycle_idx], label=model_name, linestyle='--', color=colors[model_name], alpha=0.85)
    ax1.set_title('Predicción del Uso de Arquetipo: Cycle/Control Rápido (Todos los Modelos)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Tasa de Uso', fontsize=11)
    ax1.legend(fontsize=9, loc='upper left')
    
    # Beatdown plot
    ax2.plot(x_steps, y_test_orig[:, beatdown_idx], label='Real (Histórico)', color='black', linewidth=2.2, marker='o')
    for model_name, preds in results.items():
        ax2.plot(x_steps, preds[:, beatdown_idx], label=model_name, linestyle='--', color=colors[model_name], alpha=0.85)
    ax2.set_title('Predicción del Uso de Arquetipo: Beatdown (Todos los Modelos)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Tasa de Uso', fontsize=11)
    ax2.set_xlabel('Pasos de Tiempo (Ventanas de Prueba)', fontsize=11)
    ax2.legend(fontsize=9, loc='upper left')
    
    plt.tight_layout()
    plt.savefig("chapters/images/comparacion_todos_modelos.png", dpi=300)
    plt.close()
    
    print("Finished successfully. Results saved.")

if __name__ == "__main__":
    run_experimentation()

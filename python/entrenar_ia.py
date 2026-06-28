import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set theme
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'serif'

def prepare_data():
    print("Preparing time-series data with optimized mapping...")
    data_dir = "data"
    battles_path = os.path.join(data_dir, "battlesStaging_12282020_WL_tagged.csv")
    wincons_df = pd.read_csv(os.path.join(data_dir, "Wincons.csv"))
    
    # Archetype classification wincons
    beatdown_wincons = {"Golem", "Lava Hound", "Giant", "Goblin Giant", "Electro Giant", "Elixir Golem"}
    cycle_wincons = {"Hog Rider", "Goblin Barrel", "Miner", "Wall Breakers", "Skeleton Barrel"}
    siege_wincons = {"X-Bow", "Mortar"}
    bridge_spam_wincons = {"Battle Ram", "Ram Rider", "Royal Giant"}
    control_wincons = {"Graveyard", "Three Musketeers"}
    
    archetypes = ["Beatdown", "Cycle/Control Rápido", "Asedio (Siege)", "Bridge Spam", "Control Lento", "Otros/Híbridos"]
    
    # Pre-map wincon card_id directly to archetype for O(1) lookup
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
            
    max_rows = 40000
    
    # Load only necessary columns
    cols = ['battleTime']
    for prefix in ['winner', 'loser']:
        for i in range(1, 9):
            cols.append(f'{prefix}.card{i}.id')
            
    print("Reading CSV dataset...")
    df = pd.read_csv(battles_path, usecols=cols, nrows=max_rows)
    print(f"Loaded {len(df)} rows.")
    
    # Sort chronologically by time string (fast, avoids slow pd.to_datetime parsing)
    df = df.sort_values(by='battleTime').reset_index(drop=True)
    
    # Extract archetypes using precomputed dict
    print("Mapping card IDs to archetypes...")
    winner_card_cols = [f'winner.card{i}.id' for i in range(1, 9)]
    loser_card_cols = [f'loser.card{i}.id' for i in range(1, 9)]
    
    winner_ids = df[winner_card_cols].fillna(0).astype(int).values
    loser_ids = df[loser_card_cols].fillna(0).astype(int).values
    
    winner_archs = []
    loser_archs = []
    
    for i in range(len(df)):
        # Winner
        w_arch = "Otros/Híbridos"
        for cid in winner_ids[i]:
            if cid in wincon_id_to_archetype:
                w_arch = wincon_id_to_archetype[cid]
                break
        winner_archs.append(w_arch)
        
        # Loser
        l_arch = "Otros/Híbridos"
        for cid in loser_ids[i]:
            if cid in wincon_id_to_archetype:
                l_arch = wincon_id_to_archetype[cid]
                break
        loser_archs.append(l_arch)
        
    df['winner_arch'] = winner_archs
    df['loser_arch'] = loser_archs
    
    # 100% Vectorized grouping into steps of 1000 battles
    print("Vectorizing time series generation...")
    step_size = 1000
    df['step'] = df.index // step_size
    n_steps = len(df) // step_size
    
    # Keep only full steps
    df = df[df['step'] < n_steps]
    
    w_counts = df.groupby(['step', 'winner_arch']).size().unstack(fill_value=0)
    l_counts = df.groupby(['step', 'loser_arch']).size().unstack(fill_value=0)
    
    # Align to all archetypes
    w_counts = w_counts.reindex(columns=archetypes, fill_value=0)
    l_counts = l_counts.reindex(columns=archetypes, fill_value=0)
    
    use_counts = w_counts + l_counts
    use_rates = use_counts / (step_size * 2)
    win_rates = w_counts / use_counts
    win_rates = win_rates.fillna(0.5)
    
    # Merge use rates and win rates into a single dataframe
    df_ts_list = []
    for step in range(n_steps):
        step_metrics = {}
        for arch in archetypes:
            step_metrics[f'use_{arch}'] = use_rates.loc[step, arch]
            step_metrics[f'win_{arch}'] = win_rates.loc[step, arch]
        df_ts_list.append(step_metrics)
        
    df_ts = pd.DataFrame(df_ts_list)
    print(f"Time series created with {len(df_ts)} steps.")
    return df_ts, archetypes

def train_models(df_ts, archetypes):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    # Crucial speed fix: Limit PyTorch CPU threads to 1 to avoid thread thrashing on Windows
    torch.set_num_threads(1)
    
    # Sequence creation
    seq_len = 5
    features = df_ts.columns.tolist()
    n_features = len(features)
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_ts)
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_len):
        X.append(scaled_data[i : i + seq_len])
        y.append(scaled_data[i + seq_len])
        
    X = np.array(X)
    y = np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # --- 1. LSTM Training ---
    class LSTMPredictor(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
            super(LSTMPredictor, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
            self.linear = nn.Linear(hidden_dim, output_dim)
            
        def forward(self, x):
            out, _ = self.lstm(x)
            out = self.linear(out[:, -1, :])
            return out
            
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test)
    
    model = LSTMPredictor(input_dim=n_features, hidden_dim=64, output_dim=n_features, num_layers=2)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    epochs = 200
    loss_history = []
    
    print("Training LSTM model...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss.item():.6f}", flush=True)
            
    torch.save(model.state_dict(), "python/modelo_lstm.pth")
    print("Saved LSTM model state to python/modelo_lstm.pth")
    
    # --- 2. Random Forest Regressor ---
    print("Training Random Forest baseline model...")
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_flat, y_train)
    
    with open("python/modelo_rf.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    print("Saved Random Forest model to python/modelo_rf.pkl")
    
    # --- 3. Inference and Denormalization ---
    model.eval()
    with torch.no_grad():
        lstm_preds = model(X_test_t).numpy()
        
    rf_preds = rf_model.predict(X_test_flat)
    
    y_test_orig = scaler.inverse_transform(y_test)
    lstm_preds_orig = scaler.inverse_transform(lstm_preds)
    rf_preds_orig = scaler.inverse_transform(rf_preds)
    
    metrics = []
    for idx, col in enumerate(features):
        lstm_mse = mean_squared_error(y_test_orig[:, idx], lstm_preds_orig[:, idx])
        lstm_mae = mean_absolute_error(y_test_orig[:, idx], lstm_preds_orig[:, idx])
        rf_mse = mean_squared_error(y_test_orig[:, idx], rf_preds_orig[:, idx])
        rf_mae = mean_absolute_error(y_test_orig[:, idx], rf_preds_orig[:, idx])
        
        metrics.append({
            'Variable': col,
            'LSTM_MSE': lstm_mse,
            'LSTM_MAE': lstm_mae,
            'RF_MSE': rf_mse,
            'RF_MAE': rf_mae
        })
        
    df_metrics = pd.DataFrame(metrics)
    
    print("\n=== GLOBAL METRICS COMPARISON ===")
    print(df_metrics.mean(numeric_only=True).to_markdown())
    
    # --- 4. Plot Loss History ---
    plt.figure(figsize=(6, 3.5))
    plt.plot(loss_history, color='#1f77b4', linewidth=2, label='Pérdida de Entrenamiento (MSE)')
    plt.title('Curva de Aprendizaje del Modelo LSTM', fontsize=12, fontweight='bold', pad=10)
    plt.xlabel('Época', fontsize=10)
    plt.ylabel('Pérdida (MSE)', fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig("chapters/images/lstm_loss.png", dpi=300)
    plt.close()
    
    # --- 5. Plot Actual vs Predicted ---
    cycle_idx = features.index('use_Cycle/Control Rápido')
    beatdown_idx = features.index('use_Beatdown')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    
    ax1.plot(y_test_orig[:, cycle_idx], label='Real (Histórico)', color='black', linewidth=1.8, marker='o', markersize=4)
    ax1.plot(lstm_preds_orig[:, cycle_idx], label='Predicción LSTM', color='#2ca02c', linestyle='--', linewidth=1.8)
    ax1.plot(rf_preds_orig[:, cycle_idx], label='Predicción Random Forest', color='#d62728', linestyle=':', linewidth=1.8)
    ax1.set_title('Predicción del Uso de Arquetipo: Cycle/Control Rápido', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Tasa de Uso', fontsize=10)
    ax1.legend(fontsize=9)
    
    ax2.plot(y_test_orig[:, beatdown_idx], label='Real (Histórico)', color='black', linewidth=1.8, marker='o', markersize=4)
    ax2.plot(lstm_preds_orig[:, beatdown_idx], label='Predicción LSTM', color='#2ca02c', linestyle='--', linewidth=1.8)
    ax2.plot(rf_preds_orig[:, beatdown_idx], label='Predicción Random Forest', color='#d62728', linestyle=':', linewidth=1.8)
    ax2.set_title('Predicción del Uso de Arquetipo: Beatdown', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Tasa de Uso', fontsize=10)
    ax2.set_xlabel('Pasos de Tiempo (Ventanas de Prueba)', fontsize=10)
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig("chapters/images/comparacion_modelos.png", dpi=300)
    plt.close()
    
    print("Plots saved successfully.")
    return df_metrics

if __name__ == "__main__":
    df_ts, archetypes = prepare_data()
    metrics = train_models(df_ts, archetypes)
    metrics.to_csv("data/model_comparison_metrics.csv", index=False)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
import torch.optim as optim

torch.set_num_threads(1)

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out[:, -1, :])
        return out

def main():
    df_ts = pd.read_csv("data/metagame_time_series.csv")
    archetypes = [c for c in df_ts.columns if c != 'ds']
    n_features = len(archetypes)
    
    seq_len = 5
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df_ts[archetypes])
    
    X, y = [], []
    for i in range(len(scaled_data) - seq_len):
        X.append(scaled_data[i : i + seq_len])
        y.append(scaled_data[i + seq_len])
    X, y = np.array(X), np.array(y)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    y_test_orig = scaler.inverse_transform(y_test)
    
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    print("Entrenando Random Forest...")
    rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=1)
    rf.fit(X_train_flat, y_train)
    rf_preds = scaler.inverse_transform(rf.predict(X_test_flat))
    
    print("Entrenando LSTM...")
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
        lstm_preds = scaler.inverse_transform(lstm_model(X_test_t).numpy())
    
    errors_lstm = np.mean(np.abs(y_test_orig - lstm_preds), axis=1)
    errors_rf = np.mean(np.abs(y_test_orig - rf_preds), axis=1)
    
    t_stat, p_value = stats.ttest_rel(errors_lstm, errors_rf)
    
    print("\n" + "="*60)
    print("PRUEBA DE SIGNIFICANCIA ESTADISTICA (t de Student pareada)")
    print("="*60)
    print(f"Muestras en test:   {len(errors_lstm)}")
    print(f"Media error LSTM:   {np.mean(errors_lstm):.6f}")
    print(f"Media error RF:     {np.mean(errors_rf):.6f}")
    print(f"Diferencia (RF-LSTM): {np.mean(errors_rf) - np.mean(errors_lstm):.6f}")
    print(f"Estadistico t:      {t_stat:.4f}")
    print(f"Valor p:            {p_value:.6f}")
    print(f"Nivel de confianza: {(1 - p_value) * 100:.2f}%")
    
    if p_value < 0.05:
        print("\nRESULTADO: La diferencia SI es estadisticamente significativa (p < 0.05)")
    else:
        print("\nRESULTADO: La diferencia NO es estadisticamente significativa (p >= 0.05)")
    print("="*60)

if __name__ == "__main__":
    main()

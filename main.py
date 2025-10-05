# dependencies
import pandas as pd     # data set
import numpy as np      # working with arrays
import yfinance as yf
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import root_mean_squared_error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ticker = 'AAPL'
df = yf.download(ticker, '2020-01-01')
print(df.head())

close = df['Close']

close.plot()
# plt.title(f'{ticker} Closing Prices')
# plt.xlabel('Date')
# plt.ylabel('Price ($)')
# plt.show()

scaler = StandardScaler()
scaled_close = scaler.fit_transform(df['Close'].values.reshape(-1, 1)).flatten()

seq_length = 30
data = []

for i in range(len(scaled_close) - seq_length):
    data.append(scaled_close[i:i + seq_length + 1])  # +1 for target
    
data = np.array(data)

train_size = int(0.8 * len(data))

x_train = torch.from_numpy(data[:train_size, :-1]).type(torch.Tensor).unsqueeze(-1).to(device)  # Shape: (batch, seq, 1)
y_train = torch.from_numpy(data[:train_size, -1]).type(torch.Tensor).to(device)

# slice notation for test data, not single index
x_test = torch.from_numpy(data[train_size:, :-1]).type(torch.Tensor).unsqueeze(-1).to(device)
y_test = torch.from_numpy(data[train_size:, -1]).type(torch.Tensor).to(device)

print("Training data shape:", x_train.shape)
print("Training target shape:", y_train.shape)
print("Test data shape:", x_test.shape)
print("Test target shape:", y_test.shape)

class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        
        return out
    
model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 200

for i in range(num_epochs):
    model.train()
    y_train_pred = model(x_train)
    
    # Match tensor dimensions for loss calculation
    loss = criterion(y_train_pred.squeeze(), y_train)
    
    if i % 25 == 0:
        print(f'Epoch {i}, Loss: {loss.item():.6f}')
        
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()

# Use torch.no_grad() for evaluation
with torch.no_grad():
    y_train_pred = model(x_train)
    y_test_pred = model(x_test)

# Proper inverse transformation
y_train_pred_np = y_train_pred.detach().cpu().numpy().reshape(-1, 1)
y_train_np = y_train.detach().cpu().numpy().reshape(-1, 1)
y_test_pred_np = y_test_pred.detach().cpu().numpy().reshape(-1, 1)
y_test_np = y_test.detach().cpu().numpy().reshape(-1, 1)

y_train_pred_original = scaler.inverse_transform(y_train_pred_np)
y_train_original = scaler.inverse_transform(y_train_np)
y_test_pred_original = scaler.inverse_transform(y_test_pred_np)
y_test_original = scaler.inverse_transform(y_test_np)

# RMSE calculations
train_rmse = root_mean_squared_error(y_train_original[:, 0], y_train_pred_original[:, 0])
test_rmse = root_mean_squared_error(y_test_original[:, 0], y_test_pred_original[:, 0])

print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')


plt.figure(figsize=(12, 6))
plt.plot(y_test_original, label='Actual', color='blue', alpha=0.7)
plt.plot(y_test_pred_original, label='Predicted', color='red', alpha=0.7)
plt.title(f'{ticker} Stock Price Prediction - Test Set')
plt.xlabel('Time Steps')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True)
plt.show()

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(4, 1)

ax2 = fig.add_subplot(gs[3, 0])
ax2.axhline(test_rmse, color='blue', linestyle='--', label='RMSE')


start_date_idx = train_size + seq_length
test_dates = df.index[start_date_idx:start_date_idx + len(y_test_original)]

prediction_errors = np.abs(y_test_original.flatten() - y_test_pred_original.flatten())

ax2.plot(test_dates, prediction_errors, 'r', label='Prediction Error')
ax2.legend()
ax2.set_title('Prediction Error')
ax2.set_xlabel('Date')
ax2.set_ylabel('Error')
plt.tight_layout()
plt.show()

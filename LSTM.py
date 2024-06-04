import pandas_gbq
import pandas as pd
import numpy as np
from google.oauth2 import service_account
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Assign variables for GCP project and credentials
gcp_project_id = "dce-gcp-training"

credentials = service_account.Credentials.from_service_account_file(
    '.config/gcp_service_account.json',
)

# Load data from BigQuery
data = pandas_gbq.read_gbq("SELECT * FROM `dce-gcp-training.idp_demand_forecasting.model_features`", project_id=gcp_project_id, credentials=credentials)

# Convert date string to actual date datatype
data['ds'] = pd.to_datetime(data['ds'])
data.set_index('ds', inplace=True)

# One-hot encode categorical features
data = pd.get_dummies(data, columns=['distribution_center_name', 'product_name'])

# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 30  # Number of time steps to look back
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(y.shape[1]))

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Make predictions
predictions = model.predict(X_test)

# Rescale the predictions back to the original scale
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)

# Calculate performance metrics (e.g., MAE, RMSE)
mae = np.mean(np.abs(predictions_rescaled - y_test_rescaled))
rmse = np.sqrt(np.mean((predictions_rescaled - y_test_rescaled)**2))
print(f"MAE: {mae}, RMSE: {rmse}")

# Plot predictions vs actual values
plt.figure(figsize=(14, 5))
plt.plot(y_test_rescaled[:, 0], label='Actual')
plt.plot(predictions_rescaled[:, 0], label='Predicted')
plt.legend()
plt.show()

# Create future dates for forecasting
future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=30)

# Prepare the future input data
last_sequence = scaled_data[-SEQ_LENGTH:]
future_predictions = []

for _ in range(30):
    pred = model.predict(last_sequence[np.newaxis, :, :])
    future_predictions.append(pred[0])
    last_sequence = np.append(last_sequence[1:], pred, axis=0)

future_predictions_rescaled = scaler.inverse_transform(future_predictions)

# Combine future predictions with future dates
future_forecast = pd.DataFrame(future_predictions_rescaled, index=future_dates, columns=data.columns)

# Push results to BigQuery
# pandas_gbq.to_gbq(
#     future_forecast.reset_index().rename(columns={'index': 'ds'}),
#     'dce-gcp-training.idp_demand_forecasting.lstm_model_forecast_results',
#     project_id=gcp_project_id,
#     if_exists='replace'
# )

# Save metrics to BigQuery
# metrics = pd.DataFrame({'MAE': [mae], 'RMSE': [rmse]})
# pandas_gbq.to_gbq(
#     metrics,
#     'dce-gcp-training.idp_demand_forecasting.lstm_model_metrics',
#     project_id=gcp_project_id,
#     if_exists='replace'
# )

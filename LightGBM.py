import pandas_gbq
import pandas as pd
import numpy as np
from google.oauth2 import service_account
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
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

# Create sequences for LightGBM
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i + seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

SEQ_LENGTH = 30  # Number of time steps to look back
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Flatten the sequences for LightGBM
X_flat = X.reshape(X.shape[0], -1)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, random_state=42)

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

# Define parameters for LightGBM
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train the model
bst = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[train_data, test_data], early_stopping_rounds=50)

# Make predictions
predictions = bst.predict(X_test, num_iteration=bst.best_iteration)

# Rescale the predictions back to the original scale
predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate performance metrics
mae = mean_absolute_error(y_test_rescaled, predictions_rescaled)
rmse = mean_squared_error(y_test_rescaled, predictions_rescaled, squared=False)
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
    last_sequence_flat = last_sequence.reshape(1, -1)
    pred = bst.predict(last_sequence_flat, num_iteration=bst.best_iteration)
    future_predictions.append(pred[0])
    last_sequence = np.append(last_sequence[1:], pred, axis=0)

future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Combine future predictions with future dates
future_forecast = pd.DataFrame(future_predictions_rescaled, index=future_dates, columns=['prediction'])

# Push results to BigQuery
# pandas_gbq.to_gbq(
#     future_forecast.reset_index().rename(columns={'index': 'ds'}),
#     'dce-gcp-training.idp_demand_forecasting.lightgbm_model_forecast_results',
#     project_id=gcp_project_id,
#     if_exists='replace'
# )

# Save metrics to BigQuery
# metrics = pd.DataFrame({'MAE': [mae], 'RMSE': [rmse]})
# pandas_gbq.to_gbq(
#     metrics,
#     'dce-gcp-training.idp_demand_forecasting.lightgbm_model_metrics',
#     project_id=gcp_project_id,
#     if_exists='replace'
# )

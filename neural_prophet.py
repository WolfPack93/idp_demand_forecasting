import pandas_gbq
import pandas as pd
import matplotlib.pyplot as plt
from neuralprophet import NeuralProphet, set_random_seed
from google.oauth2 import service_account

# Set random seed for reproducibility
set_random_seed(42)

# Assign variables for GCP project and credentials
gcp_project_id = "dce-gcp-training"

credentials = service_account.Credentials.from_service_account_file(
    '.config/gcp_service_account.json',
)

# Load data from BigQuery
data = pandas_gbq.read_gbq("SELECT * FROM `dce-gcp-training.idp_demand_forecasting.model_features`", project_id=gcp_project_id, credentials=credentials)

# Convert date string to actual date datatype
data['ds'] = pd.to_datetime(data['ds'])

# One-hot encode categorical regressors
# data = pd.get_dummies(data, columns=['distribution_center_name', 'product_name'])

# Create NeuralProphet model
model = NeuralProphet(
    yearly_seasonality=True,
    n_forecasts=30,  # Number of steps to forecast
    changepoints_range=0.9,
    trend_reg=0.1,
    seasonality_reg=0.1,
    n_lags=0,  # No autoregression
)

# Add regressors
for col in data.columns:
    if col.startswith('distribution_center_name_') or col.startswith('product_name_'):
        model.add_future_regressor(col)

# Fit model
metrics = model.fit(data)

# Make future dates DataFrame for forecasting
future = model.make_future_dataframe(data, periods=30, n_historic_predictions=len(data))  # Adjust forecast period as needed

# Identify the one-hot encoded columns
regressor_columns = [col for col in data.columns if col.startswith('distribution_center_name_') or col.startswith('product_name_')]

# Get the last known values of the regressors
last_known_values = data.iloc[-1][regressor_columns]

# Add regressors to the future dataframe
future = future.merge(data[['ds'] + regressor_columns], on='ds', how='left')

# Fill NaN values with the last known values
future[regressor_columns] = future[regressor_columns].fillna(last_known_values)

# Generate forecast for future dates
forecast = model.predict(future)

# Plot forecast components
model.plot_components(forecast)
plt.show()

# Output forecast data to a CSV file
# forecast.to_csv('forecast_results.csv', index=False)

########## Push results to BigQuery ##########

# Forecast results
# pandas_gbq.to_gbq(
#     forecast, 'dce-gcp-training.idp_demand_forecasting.neuralprophet_model_forecast_results', project_id=gcp_project_id, if_exists='replace',
# )

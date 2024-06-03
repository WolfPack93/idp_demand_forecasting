import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from google.oauth2 import service_account
import pandas_gbq

gcp_project_id = "dce-gcp-training"

credentials = service_account.Credentials.from_service_account_file(
    '.config/gcp_service_account.json',
)

# Load csv file
data = pandas_gbq.read_gbq("SELECT * FROM `dce-gcp-training.idp_demand_forecasting.model_features`", project_id=gcp_project_id, credentials=credentials)

# Initialize an empty DataFrame to store forecast data
forecast_data = pd.DataFrame()

# Initialize an empty DataFrame to store cross-validation results
cv_results_all = pd.DataFrame()

# Iterate over each distribution center
for center_id, group in data.groupby('distribution_center_id'):
    # 'ds' is your date column and 'y' is your demand column
    # group = group.rename(columns={'ds': 'ds', 'y': 'y'})

    # Create and fit Prophet model
    model = Prophet(yearly_seasonality=True
                    , daily_seasonality=False)
    model.fit(group)

    # Make future dates DataFrame for forecasting
    future = model.make_future_dataframe(periods=30)  # Adjust forecast period as needed

    # Generate forecast for future dates
    forecast = model.predict(future)

    # Add distribution_center_id and inventory_item_id column to the forecast DataFrame
    forecast['distribution_center_id'] = center_id
    data['ds'] = pd.to_datetime(data['ds'])
    forecast = forecast.merge(data, on=['distribution_center_id', 'ds'], how='left')
    # print(forecast)

    forecast.rename(columns={'ds_x': 'ds'}, inplace=True)
    # Concatenate the current forecast with existing forecast_data
    forecast_data = pd.concat(
        [forecast_data, forecast[['ds', 'distribution_center_id', 'product_id', 'yhat', 'yhat_lower', 'yhat_upper']]])

    # Perform cross-validation for the current distribution center
    cv_results = cross_validation(model, initial='365 days', period='180 days', horizon='30 days')

    # Add distribution_center_id to the cross-validation results
    cv_results['distribution_center_id'] = center_id

    # Concatenate the current cross-validation results with existing cv_results_all
    cv_results_all = pd.concat([cv_results_all, cv_results])

# Reset index
forecast_data.reset_index(drop=True, inplace=True)
cv_results_all.reset_index(drop=True, inplace=True)

# Output forecast data to a CSV file
# forecast_data.to_csv('forecast_results.csv', index=False)

# Example usage:
print(forecast_data)

# Calculate error metrics for all cross-validation results
metrics = performance_metrics(cv_results_all)

# Print error metrics
print(metrics)

# Visualize results
# plot_cross_validation_metric(cv_results_all, metric='mae')
# plt.show()

# Push results to BigQuery

# Forecast results
pandas_gbq.to_gbq(
    forecast_data, 'dce-gcp-training.idp_demand_forecasting.prophet_model_forecast_results', project_id=gcp_project_id, if_exists='replace',
)

# Model metrics
metrics['horizon'] = metrics['horizon'].astype(str)
pandas_gbq.to_gbq(
    metrics, 'dce-gcp-training.idp_demand_forecasting.prophet_model_metrics', project_id=gcp_project_id, if_exists='replace',
)

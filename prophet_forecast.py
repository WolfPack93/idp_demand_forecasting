import warnings
import pandas_gbq
import pandas as pd
import itertools
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from google.oauth2 import service_account

warnings.simplefilter(action='ignore', category=FutureWarning)


# Function to sanitize column names for BigQuery
def sanitize_column_names(df):
    df.columns = df.columns.str.replace(r'[^a-zA-Z0-9_]', '_', regex=True)
    df.columns = df.columns.str.replace(r'__+', '_', regex=True)
    df.columns = df.columns.str.strip('_')
    return df


# Function to revert one-hot encoding
def undummy(df, prefix):
    dummy_cols = [col for col in df.columns if col.startswith(prefix)]
    df[prefix.rstrip('_')] = df[dummy_cols].idxmax(axis=1).str.replace(prefix, '')
    df.drop(columns=dummy_cols, inplace=True)
    return df


# Assign variables for GCP project and credentials
gcp_project_id = "dce-gcp-training"

credentials = service_account.Credentials.from_service_account_file(
    '.config/gcp_service_account.json',
)

# Load data from BigQuery
data = pandas_gbq.read_gbq("SELECT * FROM `dce-gcp-training.idp_demand_forecasting.model_features`",
                           project_id=gcp_project_id, credentials=credentials)

# Convert date string to actual date datatype
data['ds'] = pd.to_datetime(data['ds'])

# Prepare to store results
all_forecasts = []
# all_metrics = []

# Iterate over unique combinations of distribution_center_name and product_name
unique_combinations = data.groupby(['ds', 'distribution_center_name', 'product_name']).size().reset_index().drop(0, axis=1)

for index, row in unique_combinations.iterrows():
    distribution_center = row['distribution_center_name']
    product = row['product_name']

    # Filter data for the current combination
    subset_data = data[(data['distribution_center_name'] == distribution_center) & (data['product_name'] == product)]

    # One-hot encode categorical regressors
    subset_data = pd.get_dummies(subset_data, columns=['distribution_center_name', 'product_name'])

    # Create Prophet model
    model = Prophet(
        yearly_seasonality=True,
        changepoint_prior_scale=0.1,
        seasonality_prior_scale=0.01
    )

    # Add regressors
    for col in subset_data.columns:
        if col.startswith('distribution_center_name_') or col.startswith('product_name_'):
            model.add_regressor(col)

    # Fit model
    model.fit(subset_data)

    # Make future dates DataFrame for forecasting
    future = model.make_future_dataframe(periods=30)  # Adjust forecast period as needed

    # Identify the one-hot encoded columns
    regressor_columns = [col for col in subset_data.columns if
                         col.startswith('distribution_center_name_') or col.startswith('product_name_')]

    # Get the last known values of the regressors
    last_known_values = subset_data.iloc[-1][regressor_columns]

    # Add regressors to the future dataframe
    future = future.merge(subset_data[['ds'] + regressor_columns], on='ds', how='left')

    # Fill NaN values with the last known values
    future[regressor_columns] = future[regressor_columns].fillna(last_known_values)

    # Generate forecast for future dates
    forecast = model.predict(future)

    # Merge one-hot encoded regressor columns from future into forecast based on 'ds'
    # forecast = forecast.merge(future[['ds'] + regressor_columns], on='ds', how='left')

    # Undummy 'distribution_center_name'
    forecast = undummy(forecast, 'distribution_center_name_')

    # Undummy 'product_name'
    forecast = undummy(forecast, 'product_name_')

    # Keep only necessary columns
    forecast = forecast[['ds', 'distribution_center_name', 'product_name', 'yhat', 'yhat_lower', 'yhat_upper']]

    # Append forecast to all_forecasts
    all_forecasts.append(forecast)

    # # Perform cross-validation
    # cv_results = cross_validation(model, initial='367 days', period='60 days', horizon='10 days')
    # metrics = performance_metrics(cv_results)
    #
    # # Print cv results and metrics for each combination
    # print(f"== Cross-validation Results for {distribution_center} - {product}:")
    # print(cv_results)
    # print(f"== Performance Metrics for {distribution_center} - {product}:")
    # print(metrics)
    #
    # metrics['distribution_center_name'] = distribution_center
    # metrics['product_name'] = product
    # metrics['horizon'] = metrics['horizon'].astype(str)
    #
    # all_metrics.append(metrics)

# Combine all forecasts and metrics into a single DataFrame
forecast_data = pd.concat(all_forecasts)
# metric_data = pd.concat(all_metrics)

# Check the results
print(forecast_data.head())

# Output forecast data to a CSV file
# forecast_data.to_csv('forecast_results.csv', index=False)

# Push forecast results to BigQuery
pandas_gbq.to_gbq(
    forecast_data,
    'dce-gcp-training.idp_demand_forecasting.prophet_model_forecast_results',
    project_id=gcp_project_id,
    if_exists='replace',
    credentials=credentials
)

# Save metrics to BigQuery
# pandas_gbq.to_gbq(
#         metric_data,
#         'dce-gcp-training.idp_demand_forecasting.prophet_model_metrics',
#         project_id=gcp_project_id,
#         if_exists='append',
#         credentials=credentials
#     )

# Optionally, you can perform cross-validation and calculate metrics for each model
# Iterate over unique combinations again
# for index, row in unique_combinations.iterrows():
#     distribution_center = row['distribution_center_name']
#     product = row['product_name']
#
#     # Filter data for the current combination
#     subset_data = data[(data['distribution_center_name'] == distribution_center) & (data['product_name'] == product)]
#
#     # Create and fit Prophet model as before
#     model = Prophet(
#         yearly_seasonality=True,
#         changepoint_prior_scale=0.1,
#         seasonality_prior_scale=0.01
#     )
#
#     for col in subset_data.columns:
#         if col.startswith('distribution_center_name_') or col.startswith('product_name_'):
#             model.add_regressor(col)
#
#     model.fit(subset_data)



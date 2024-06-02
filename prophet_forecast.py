import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric

# Load csv file
data = pd.read_csv('model_features.csv')

# Initialize an empty DataFrame to store forecast data
forecast_data = pd.DataFrame()

# Initialize an empty DataFrame to store cross-validation results
cv_results_all = pd.DataFrame()

# Iterate over each distribution center
for center_id, group in data.groupby('distribution_center_id'):
    # 'ds' is your date column and 'y' is your demand column
    group = group.rename(columns={'ds': 'ds', 'y': 'y'})

    # Create and fit Prophet model
    model = Prophet(yearly_seasonality=True, daily_seasonality=True)
    model.fit(group)

    # Make future dates DataFrame for forecasting
    future = model.make_future_dataframe(periods=30)  # Adjust forecast period as needed

    # Generate forecast for future dates
    forecast = model.predict(future)

    # Add distribution_center_id and inventory_item_id column to the forecast DataFrame
    forecast['distribution_center_id'] = center_id
    forecast['inventory_item_id'] = data['inventory_item_id']

    # Concatenate the current forecast with existing forecast_data
    forecast_data = pd.concat(
        [forecast_data, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'distribution_center_id', 'inventory_item_id']]])

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
forecast_data.to_csv('forecast_results.csv', index=False)

# Example usage:
print(forecast_data)

# Calculate error metrics for all cross-validation results
metrics = performance_metrics(cv_results_all)

# Print error metrics
print(metrics)

# Visualize results
plot_cross_validation_metric(cv_results_all, metric='mae')
plt.show()

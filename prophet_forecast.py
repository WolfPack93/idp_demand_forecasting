import pandas as pd
from prophet import Prophet

# Load csv file
data = pd.read_csv('model_features.csv')

# Initialize an empty DataFrame to store forecast data
forecast_data = pd.DataFrame()

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

# Reset index
forecast_data.reset_index(drop=True, inplace=True)

# Output forecast data to a CSV file
forecast_data.to_csv('forecast_results.csv', index=False)

# Example usage:
print(forecast_data)

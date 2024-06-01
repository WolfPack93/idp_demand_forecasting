# Purpose: A python script to run a Prophet Time Series ML prediction model for ecommerce demand forecasting
# Creator: Ed Wolf
# Created: 20240601

import pandas as pd
from prophet import Prophet

version = '0.1.0'


def predict(csv_file_path):
    # Load data from CSV file
    df = pd.read_csv(csv_file_path)

    # Rename columns as required by Prophet
    df.rename(columns={'ds': 'ds', 'y': 'y', 'distribution_center_id': 'distribution_center_id',
                       'inventory_item_id': 'inventory_item_id'}, inplace=True)

    # Convert 'ds' column to proper datetime objects
    df['ds'] = pd.to_datetime(df['ds'])

    # Initialize an empty dataframe for the forecast
    forecast_df = pd.DataFrame()

    # Group the data by 'distribution_center_id' and 'inventory_item_id'
    grouped = df.groupby(['distribution_center_id', 'inventory_item_id'])

    # Loop through each group
    for (dc_id, item_id), group_df in grouped:
        # Initialize the Prophet model
        m = Prophet()

        # Add external regressors
        m.add_regressor('distribution_center_id')
        m.add_regressor('inventory_item_id')

        # Fit the model
        m.fit(group_df)

        # Create a future dataframe for the next 30 days
        future = m.make_future_dataframe(periods=30)

        # Add the same regressors to the future dataframe
        future['distribution_center_id'] = dc_id
        future['inventory_item_id'] = item_id

        # Make predictions
        forecast = m.predict(future)

        # Add 'distribution_center_id' and 'inventory_item_id' columns to the forecast
        forecast['distribution_center_id'] = dc_id
        forecast['inventory_item_id'] = item_id

        # Append the forecast to the main forecast dataframe
        forecast_df = pd.concat([forecast_df, forecast], ignore_index=True)

    # Return the forecast dataframe
    return forecast_df


def main():
    # Define the path to the CSV file
    csv_file_path = r'path_to_csv'

    # Run the prediction
    forecast = predict(csv_file_path)

    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'distribution_center_id', 'inventory_item_id']])

    # Further processing can be done with the forecast here if needed


if __name__ == "__main__":
    main()

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

    # Initialize the Prophet model
    m = Prophet()

    # Add external regressors
    m.add_regressor('distribution_center_id')
    m.add_regressor('inventory_item_id')

    # Fit the model
    m.fit(df)

    # Create a future dataframe for the next 30 days
    future = m.make_future_dataframe(periods=30)

    # Add the same regressors to the future dataframe (here you might want to fill with reasonable future values)
    future['distribution_center_id'] = df['distribution_center_id'].iloc[-1]
    future['inventory_item_id'] = df['inventory_item_id'].iloc[-1]

    # Make predictions
    forecast = m.predict(future)

    # Return the forecast
    return forecast


def main():
    # Define the path to the CSV file
    csv_file_path = r'C:\Users\edward.wolf\Downloads\model_features.csv'

    # Run the prediction
    forecast = predict(csv_file_path)

    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])

    # Further processing can be done with the forecast here if needed


if __name__ == "__main__":
    main()

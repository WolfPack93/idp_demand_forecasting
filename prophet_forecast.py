# Purpose: A python script to run a Prophet Time Series ML prediction model for ecommerce demand forecasting
# Creator: Ed Wolf
# Created: 20240601

import pandas as pd
from prophet import Prophet

version = '0.1.0'

def predict(csv_file_path):
    # Load data from CSV file
    df = pd.read_csv(csv_file_path)

    # Print the first few rows of the dataframe
    print("== Initial DataFrame:")
    print(df.head())

    # Rename columns as required by Prophet
    df.rename(columns={'ds': 'ds', 'y': 'y', 'distribution_center_id': 'distribution_center_id',
                       'inventory_item_id': 'inventory_item_id'}, inplace=True)

    # Convert 'ds' column to proper datetime objects
    df['ds'] = pd.to_datetime(df['ds'])

    # Check for any NaN values in the dataframe
    print("== NaN values in DataFrame:")
    print(df.isna().sum())

    # Initialize an empty DataFrame to store all forecasts
    all_forecasts = pd.DataFrame()

    # Get unique combinations of distribution_center_id and inventory_item_id
    unique_combinations = df[['distribution_center_id', 'inventory_item_id']].drop_duplicates()

    for _, row in unique_combinations.iterrows():
        distribution_center_id = row['distribution_center_id']
        inventory_item_id = row['inventory_item_id']

        # Filter the dataframe for each combination
        df_comb = df[(df['distribution_center_id'] == 1)]

        # print("== df comb:")
        # print(df_comb)

        # Debugging print to check the filtered data
        # print(f"\nFiltered DataFrame for distribution_center_id={distribution_center_id} and inventory_item_id={inventory_item_id}:")
        # print(df_comb.head())

        # Initialize the Prophet model
        m = Prophet()

        # Fit the model
        m.fit(df_comb[['ds', 'y']])

        # Create a future dataframe for the next 30 days
        future = m.make_future_dataframe(periods=7)

        # Make predictions
        forecast = m.predict(future)

        # Add the combination columns to the forecast
        forecast['distribution_center_id'] = distribution_center_id
        forecast['inventory_item_id'] = inventory_item_id

        # Append the forecast to all_forecasts DataFrame
        all_forecasts = pd.concat([all_forecasts, forecast], ignore_index=True)

    # Return the forecast
    return all_forecasts

def main():
    # Define the path to the CSV file
    csv_file_path = r'path_to_csv'

    # Run the prediction
    forecast = predict(csv_file_path)

    # Print the first 5 rows of the forecast
    print(forecast.head())

    # Further processing can be done with the forecast here if needed

if __name__ == "__main__":
    main()

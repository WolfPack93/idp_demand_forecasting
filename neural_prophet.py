import pandas as pd
from neuralprophet import NeuralProphet

# Load data
df = pd.read_csv('path_to_csv')

# Convert 'ds' column to datetime
df['ds'] = pd.to_datetime(df['ds'])


# Function to train and forecast using NeuralProphet for each combination of distribution_center_id and inventory_item_id
def train_and_forecast(df):
    results = []

    # Get unique combinations of distribution_center_id and inventory_item_id
    combinations = df[['distribution_center_id', 'inventory_item_id']].drop_duplicates()

    for _, combo in combinations.iterrows():
        dc_id = combo['distribution_center_id']
        item_id = combo['inventory_item_id']

        # Filter the dataframe for each combination
        df_comb = df[(df['distribution_center_id'] == dc_id) &
                     (df['inventory_item_id'] == item_id)]

        # Initialize the NeuralProphet model
        model = NeuralProphet()

        # Fit the model
        model.fit(df_comb, freq='D')

        # Make future dataframe
        future = model.make_future_dataframe(df_comb, periods=30)

        # Forecast
        forecast = model.predict(future)

        # Store the result
        forecast['distribution_center_id'] = dc_id
        forecast['inventory_item_id'] = item_id
        results.append(forecast)

    # Concatenate all results
    final_forecast = pd.concat(results, ignore_index=True)

    return final_forecast


# Train and forecast
forecast = train_and_forecast(df)

# Check the forecast
print(forecast[['ds', 'yhat1', 'yhat_lower1', 'yhat_upper1', 'distribution_center_id', 'inventory_item_id']].head())

# Save the forecast to a CSV file
forecast.to_csv('forecast_results.csv', index=False)

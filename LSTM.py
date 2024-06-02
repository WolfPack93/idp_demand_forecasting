import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('C:/Users/edward.wolf/Documents/idp_projects/idp_demand_forecasting_prophet_model/model_features.csv')

# Convert 'ds' column to datetime
df['ds'] = pd.to_datetime(df['ds'])

# Encode 'distribution_center_id' and 'inventory_item_id'
encoder_dc = LabelEncoder()
encoder_item = LabelEncoder()
df['distribution_center_id'] = encoder_dc.fit_transform(df['distribution_center_id'])
df['inventory_item_id'] = encoder_item.fit_transform(df['inventory_item_id'])

# Normalize 'y'
scaler = MinMaxScaler()
df['y'] = scaler.fit_transform(df['y'].values.reshape(-1, 1))

# Sort data by 'ds'
df = df.sort_values(by='ds')

# Prepare data for LSTM
def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
    return np.array(sequences)

seq_length = 10
X = []
y = []
for dist_center in df['distribution_center_id'].unique():
    for item in df['inventory_item_id'].unique():
        df_filtered = df[(df['distribution_center_id'] == dist_center) & (df['inventory_item_id'] == item)]
        if len(df_filtered) > seq_length:
            sequences = create_sequences(df_filtered['y'].values, seq_length)
            for seq in sequences:
                X.append(seq[:-1])
                y.append(seq[-1])

X = np.array(X)
y = np.array(y)

# Reshape X to be suitable for LSTM (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length - 1, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Print model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}')

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Make predictions
y_pred = model.predict(X_test)

# Inverse transform the predictions
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Compare predictions to actual values
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()

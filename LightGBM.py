import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import numpy as np

# Load data
df = pd.read_csv('path_to_csv')

# Convert 'ds' column to datetime
df['ds'] = pd.to_datetime(df['ds'])

# Encode categorical variables
# encoder_dc = LabelEncoder()
# encoder_item = LabelEncoder()
# df['distribution_center_id'] = encoder_dc.fit_transform(df['distribution_center_id'])
# df['inventory_item_id'] = encoder_item.fit_transform(df['inventory_item_id'])

# Split data into features and target
X = df.drop(['y'], axis=1)
y = df['y']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert datetime columns to Unix timestamp format
X_train['ds'] = pd.to_datetime(X_train['ds']).astype('int64') // 10**9  # Convert to seconds since epoch
X_test['ds'] = pd.to_datetime(X_test['ds']).astype('int64') // 10**9  # Convert to seconds since epoch

# Create LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

# Set parameters for LightGBM classifier
params = {
    'objective': 'multiclass',
    'num_class': 11, #len(y),  # Number of classes
    'metric': 'multi_logloss'
}

# Train the model
num_round = 10000
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])

# Predictions
y_pred = bst.predict(X_test)

# Convert predictions to class labels
y_pred_labels = [np.argmax(pred) for pred in y_pred]

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred_labels)
print("Accuracy:", accuracy)

# Example prediction
example_data = X_test
example_data['ds'] = pd.to_datetime('2024-06-01')  # Replace with your desired date
example_data['ds'] = example_data['ds'].astype('int64') // 10**9  # Convert to seconds since epoch
example_pred = bst.predict(example_data)

# Convert prediction to class label
example_pred_label = np.argmax(example_pred)
print("Predicted class label:", example_pred_label)

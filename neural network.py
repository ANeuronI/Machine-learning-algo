import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.pipeline import make_pipeline

# Load the dataset
dataset_path = 'ml\WC_Train.csv'
cricket_data = pd.read_csv(dataset_path)

# Handle categorical features by encoding them
label_encoder = LabelEncoder()
for column in cricket_data.columns:
    if cricket_data[column].dtype == 'object':
        cricket_data[column] = label_encoder.fit_transform(cricket_data[column])

# Split the dataset into features (X) and target variables (y1, y2)
feature_columns = ['Team A', 'Team B', 'Ground']  # Adjust feature names as needed
X = cricket_data[feature_columns]
y1 = cricket_data['Won']  # Replace with actual target variable name
y2 = cricket_data['Team A Won']  # Replace with actual target variable name

# Split the data into training and testing sets
X_train, X_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    X, y1, y2, test_size=0.2, random_state=42
)

# Build a neural network model for y1
model_y1 = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42)
)

# Build a neural network model for y2
model_y2 = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=1000, random_state=42)
)

# Fit the model for y1
model_y1.fit(X_train, y1_train)

# Fit the model for y2
model_y2.fit(X_train, y2_train)

# Predict on the test set for y1
y1_pred = model_y1.predict(X_test)

# Predict on the test set for y2
y2_pred = model_y2.predict(X_test)

# Evaluate the models
mse_y1 = mean_squared_error(y1_test, y1_pred)
mse_y2 = mean_squared_error(y2_test, y2_pred)

mae_y1 = mean_absolute_error(y1_test, y1_pred)
mae_y2 = mean_absolute_error(y2_test, y2_pred)

rmse_y1 = sqrt(mse_y1)
rmse_y2 = sqrt(mse_y2)

# Print the results
print(f'Mean Squared Error (y1): {mse_y1:.2f}')
print(f'Mean Squared Error (y2): {mse_y2:.2f}')

print(f'Mean Absolute Error (y1): {mae_y1:.2f}')
print(f'Mean Absolute Error (y2): {mae_y2:.2f}')

print(f'Root Mean Squared Error (y1): {rmse_y1:.2f}')
print(f'Root Mean Squared Error (y2): {rmse_y2:.2f}')

# Set a tolerance level for predictions
tolerance = 0.5  # You can adjust this based on the scale of your target variables

# Calculate the percentage of predictions within the tolerance for y1
within_tolerance_y1 = (abs(y1_test - y1_pred) <= tolerance).mean() * 100

# Calculate the percentage of predictions within the tolerance for y2
within_tolerance_y2 = (abs(y2_test - y2_pred) <= tolerance).mean() * 100

print(f'Percentage of predictions within {tolerance} for y1: {within_tolerance_y1:.2f}%')
print(f'Percentage of predictions within {tolerance} for y2: {within_tolerance_y2:.2f}%')

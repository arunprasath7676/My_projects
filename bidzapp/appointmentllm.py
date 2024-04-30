import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import joblib

# Load appointment data 
df = pd.read_csv('D:/ticket/appointment/appointment.csv')

# Understanding data
print(df.shape)
print(df.columns)
print(df.head(5))
print(df.info())
print(df.describe())
print(df.groupby('appointment_date').size())

# Convert 'appointment_date' to datetime
df['appointment_date'] = pd.to_datetime(df['appointment_date'])

# Count appointments per date, and extract day of week and month
df['appointment_count'] = df.groupby(df['appointment_date'].dt.date)['appointment_date'].transform('count')
df = df[['appointment_date', 'appointment_count']].drop_duplicates()
df['day_of_week'] = df['appointment_date'].dt.dayofweek
df['month'] = df['appointment_date'].dt.month

# Replacing null value columns (text) with most used value
text_cols = df.select_dtypes(include=['object']).columns
print("Text Columns:", text_cols)
for col in text_cols:
    mode_value = df[col].mode().iloc[0]
    df[col] = df[col].fillna(mode_value)

# Replacing null value columns (int, float) with most used value
numeric_cols = df.select_dtypes(include=['integer', 'float']).columns
print("Numeric Columns:", numeric_cols)
for col in numeric_cols:
    mode_value_numeric = df[col].mode().iloc[0]
    df[col] = df[col].fillna(mode_value_numeric)
print("columns",df)
# Dropping unwanted columns
# df = df.drop(["id", "is_flagged", "created_at", "updated_at"], axis=1)
# print(df.shape)

print("completed")
df.to_csv('D:/ticket/appointment/preprocessing_data.csv', index=False)



df = pd.read_csv('D:/ticket/appointment/preprocessing_data.csv')
#df = pd.read_csv('./06_output_data.csv')

# Define features and target
features = ['day_of_week', 'month']  # Define your features here
target = 'appointment_count'  # Change the target variable to appointment_count

# Split the data into features (X) and target (y)
X = df[features]
y = df[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Create and train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions on the test set
y_predictions = regressor.predict(X_test)

# Calculate evaluation metrics
mean_squared_error_value = mean_squared_error(y_test, y_predictions)
root_mean_squared_error = sqrt(mean_squared_error_value)
r_squared = regressor.score(X_test, y_test)

print("Mean Squared Error:", mean_squared_error_value)
print("Root Mean Squared Error:", root_mean_squared_error)
print("R-squared:", r_squared)

# Plot results
plt.scatter(y_predictions, y_test, c='r', label='Actual vs. Predicted')
plt.plot(y_test, y_test, color='k', label='Perfect Prediction')
plt.title('Actual vs. Predicted Appointment Counts')
plt.xlabel('Predicted Counts')
plt.ylabel('Actual Counts')
plt.legend()
plt.show()

# Plot residuals
plt.scatter(y_predictions, (y_predictions - y_test), c='b')
plt.hlines(y=0, xmin=min(y_predictions), xmax=max(y_predictions), colors='k', linestyles='dashed')
plt.title('Residual Plot')
plt.xlabel('Predicted Counts')
plt.ylabel('Residuals')
plt.show()

# Save the model
joblib.dump(regressor, 'D:/ticket/appointment/appointment_count_model.pkl')


# # Load preprocessed data
# df = pd.read_csv('D:/ticket/appointment/preprocessing_data.csv')

# # Define features and target
# features = ['day_of_week', 'month']  # Define your features here
# target = 'appointment_count'  # Change the target variable to appointment_count

# # Split the data into features (X) and target (y)
# X = df[features]
# y = df[target]

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y)
# #X_train, X_test, y_train, y_test = train_test_split(X, y)

def train_and_evaluate_regressor(regressor, X_train, y_train, X_test, y_test):
    regressor.fit(X_train, y_train)
    y_predictions = regressor.predict(X_test)
    score = regressor.score(X_test, y_test)
    rmse = sqrt(mean_squared_error(y_test, y_predictions))
    return score, rmse


def linear():
    regressor = LinearRegression()
    return train_and_evaluate_regressor(regressor, X_train, y_train, X_test, y_test)

def ridge():
    regressor = Ridge(alpha=.3)
    return train_and_evaluate_regressor(regressor, X_train, y_train, X_test, y_test)

def lasso():
    regressor = Lasso(alpha=0.00009)
    return train_and_evaluate_regressor(regressor, X_train, y_train, X_test, y_test)

def elasticnet():
    regressor = ElasticNet(alpha=1, l1_ratio=0.5)
    return train_and_evaluate_regressor(regressor, X_train, y_train, X_test, y_test)

def randomforest():
    regressor = RandomForestRegressor(n_estimators=15, min_samples_split=15, criterion='squared_error', max_depth=None)
    return train_and_evaluate_regressor(regressor, X_train, y_train, X_test, y_test)

def perceptron():
    regressor = MLPRegressor(hidden_layer_sizes=(5000,), activation='relu', solver='adam', max_iter=1000)
    return train_and_evaluate_regressor(regressor, X_train, y_train, X_test, y_test)

def decisiontree():
    regressor = DecisionTreeRegressor(min_samples_split=30, max_depth=None)
    return train_and_evaluate_regressor(regressor, X_train, y_train, X_test, y_test)

def adaboost():
    regressor = AdaBoostRegressor(random_state=8, loss='exponential')
    return train_and_evaluate_regressor(regressor, X_train, y_train, X_test, y_test)

def extratrees():
    regressor = ExtraTreesRegressor(n_estimators=50)
    return train_and_evaluate_regressor(regressor, X_train, y_train, X_test, y_test)

def gradientboosting():
    regressor = GradientBoostingRegressor(loss='squared_error', n_estimators=500, min_samples_split=15)
    return train_and_evaluate_regressor(regressor, X_train, y_train, X_test, y_test)

print("Score, RMSE values")
print("Linear = ", linear())
print("Ridge = ", ridge())
print("Lasso = ", lasso())
print("ElasticNet = ", elasticnet())
print("RandomForest = ", randomforest())
print("Perceptron = ", perceptron())
print("DecisionTree = ", decisiontree())
print("AdaBoost = ", adaboost())
print("ExtraTrees = ", extratrees())
print("GradientBoosting = ", gradientboosting())


df = pd.read_csv('D:/ticket/appointment/preprocessing_data.csv')

# Define features and target
features = ['day_of_week', 'month']  # Define your features here
target = 'appointment_count'  # Change the target variable to appointment_count

# Prepare data
X = df[features]
y = df[target]

# Define and train model
# model = Pipeline([
#     ('encoder', ColumnTransformer([
#         ('onehot', OneHotEncoder(), ['day_of_week', 'month'])
#     ], remainder='passthrough')),
#     ('regressor', LinearRegression())
# ])
# model.fit(X, y)

model = Pipeline([
    ('encoder', ColumnTransformer([
        ('onehot', OneHotEncoder(handle_unknown='ignore'), ['day_of_week', 'month'])
    ], remainder='passthrough')),
    ('regressor', LinearRegression())
])
model.fit(X, y)

# Generate dates for the next 2 years
start_date = pd.to_datetime(df['appointment_date']).max() + pd.DateOffset(days=1)
end_date = start_date + pd.DateOffset(years=2)
dates_2_years = pd.date_range(start=start_date, end=end_date, freq='D')

# Extract features for the next 2 years
features_2_years = pd.DataFrame({
    'appointment_date': dates_2_years,
    'day_of_week': dates_2_years.dayofweek,
    'month': dates_2_years.month
})

# Make predictions for the next 2 years
predictions_2_years = model.predict(features_2_years[['day_of_week', 'month']])

# Visualize results
# plt.figure(figsize=(10, 6))
# plt.plot(dates_2_years, predictions_2_years, label='Predicted Appointment Counts')
# plt.title('Predicted Appointment Counts for the Next 2 Years')
# plt.xlabel('Date')
# plt.ylabel('Appointment Counts')
# plt.legend()
# plt.show()

file_path = 'D:/ticket/appointment/predicted_appointment_counts.csv'

predicted_data = pd.DataFrame({
    'Date': dates_2_years,
    'Predicted Appointment Counts': predictions_2_years
})

predicted_data['Predicted Appointment Counts'] = predicted_data['Predicted Appointment Counts'].round()
# Convert dates to string format
predicted_data['Date'] = predicted_data['Date'].dt.strftime('%Y-%m-%d')
predicted_json = predicted_data.to_json(orient='records')

print(predicted_json)

predicted_data.to_csv(file_path, index=False)

print("Predicted appointment counts saved to:", file_path)
# with open(file_path, 'w') as json_file:
#     json_file.write(predicted_json)



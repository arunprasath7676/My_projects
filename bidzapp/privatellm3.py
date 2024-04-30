import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import numpy as np

# Load the dataset
df = pd.read_csv('D:/ticket/appointment/vector.csv')

# Separate the target variable from the features
y = df['data[0]']
X = df.drop('data[0]', axis=1)

# Check for missing values 
missing_values = X.isnull().sum()
print("Missing values:\n", missing_values)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Check for infinite values
print("Is infinite:", np.isfinite(X_imputed).all())

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=100)

# Train the linear regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Evaluate the model
train_score = lr.score(X_train, y_train)
test_score = lr.score(X_test, y_test)
print("Train Score:", train_score)
print("Test Score:", test_score)

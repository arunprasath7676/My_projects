import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from math import sqrt
import matplotlib.pyplot as plt

# Define database connection parameters
db_user = "mediyoga"
db_password = "mediyoga2023"
db_host = "3.111.133.67"
db_name = "Medi_Yoga"

# Define the MySQL URI
mysql_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

# Create an SQLAlchemy engine
engine = create_engine(mysql_uri)

# Load data from each table into separate DataFrames
table_names = ['appointment', 'appointment__clinical_report', 'appointment__symptom', 'banner', 'bill', 'case',
            'case_category', 'case_sheet', 'case_sheet__diagnosis', 'case_sheet__doctor_symptom',
             'clinic', 'clinic__working_hours', 'consent_form', 'diagnosis', 'diagnosis__category', 
             'doctor_details', 'doctor_details__favorite_drug', 'drug', 'drug__template', 
             'drug__template_mapping', 'estimation', 'estimation__diagnosis', 'exercises', 
             'exercises__image', 'exercises__mistakes', 'exercises__modulations', 
             'exercises__modulations_image', 'failed_jobs', 'lab_request', 'lab_request__investigation', 
             'letter_master', 'letter_type', 'migrations', 'notification', 'otp_log', 
             'password_reset_tokens', 'patients', 'patients__consent_form', 'patients__exercise',
             'patients__exercise_modulation', 'patients__lab_requests', 'patients__lab_requests_list',
             'patients__letters', 'patients__payments', 'patients__payments_diagnosis',
             'patients__prescription', 'patients__prescription_drugs', 'patients__property',
             'patients__property_amount', 'personal_access_tokens', 'property', 'property__inventory', 
             'property__inventory_vendor', 'property__vendor', 'resource', 'symptom', 'symptom__doctor',
             'symptom__pre_visit', 'time_master', 'time_master__available', 'time_master__interval',
               'time_master__slot', 'users', 'week_slot']

preprocessed_dfs = []
for table_name in table_names:
    # Load data from each table into a DataFrame
    query = f"SELECT * FROM `{table_name}`"
    table_df = pd.read_sql(query, engine)
    
    # Drop columns with all-NA values or empty columns
    table_df = table_df.dropna(axis=1, how='all')
        
    
    # Replacing null value columns (text) with most used value
    text_cols = table_df.select_dtypes(include=['object']).columns
    for col in text_cols:
        mode_value = table_df[col].mode().iloc[0]
        table_df[col] = table_df[col].fillna(mode_value)
    
    # Replacing null value columns (int, float) with most used value
    numeric_cols = table_df.select_dtypes(include=['integer', 'float']).columns
    for col in numeric_cols:
        mode_value_numeric = table_df[col].mode().iloc[0]
        table_df[col] = table_df[col].fillna(mode_value_numeric)
    
    # Append the preprocessed DataFrame to the list
    preprocessed_dfs.append(table_df)

merged_data = pd.concat(preprocessed_dfs, ignore_index=True)

# Store the merged DataFrame to a single CSV file
merged_data.to_csv('D:/ticket/appointment/preprocessing_fulldatabase.csv', index=False)

print("Completed storing preprocessed DataFrames to a single CSV file.")


df = pd.read_csv('D:/ticket/appointment/preprocessing_fulldatabase.csv')



features = ''  
target = ''  

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
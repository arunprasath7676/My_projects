# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine

# Connect to your local MySQL database
db_user = "mediyoga"
db_password = "mediyoga2023"
db_host = "3.111.133.67"
db_name = "Medi_Yoga"

# Adjust the connection details according to your database setup
engine = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}")

# Assuming 'data_table' is the table you want to fetch data from
query = "SELECT * FROM appointment"
df = pd.read_sql(query, engine)

# Step 2: Data Preprocessing
# Drop unnecessary columns such as 'id', 'token', 'created_at', 'updated_at', 'deleted_at'
df.drop(['id', 'token', 'created_at', 'updated_at', 'deleted_at'], axis=1, inplace=True)

# Convert appointment_date to Unix timestamp
df['appointment_date'] = pd.to_datetime(df['appointment_date']).astype(int) // 10**9

# Let's say 'X' contains features and 'y' contains the target variable
X = df.drop(['appointment_status'], axis=1)  
y = df['appointment_status']  # Use 'appointment_status' as the target variable

# Step 3: Splitting the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Training
model = LogisticRegression()  # Using Logistic Regression for classification
model.fit(X_train, y_train)

# Step 5: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

# Step 6: Prediction (Assuming you want to predict future data)
# Fetching new data from the database
new_data_query = "SELECT * FROM appointment"
new_data = pd.read_sql(new_data_query, engine)

# Preprocess new_data similarly to X (drop unnecessary columns, etc.)
new_data_processed = new_data.drop(['id', 'token', 'created_at', 'updated_at', 'deleted_at'], axis=1)

# Making predictions on new data
future_predictions = model.predict(new_data_processed)

print('Future Predictions:', future_predictions)

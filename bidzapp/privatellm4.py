import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
import joblib

# Define database connection parameters
db_user = "mediyoga"
db_password = "mediyoga2023"
db_host = "3.111.133.67"
db_name = "Medi_Yoga"

# Define the MySQL URI
mysql_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

# Create an SQLAlchemy engine
engine = create_engine(mysql_uri)


def get_table_info(table_name):
    # Execute SQL query to get table information
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, con=engine)
    # Define the file path with the table name
    filename = f'D:/ticket/appointment/{table_name}_info.csv'
    # Export DataFrame to CSV
    df.to_csv(filename, index=False)
    print(f"Table information exported to {filename}")


def build_model(table_name, target):
    # Read data from CSV (assuming data is pre-exported using get_table_info)
    df = pd.read_csv(f'D:/ticket/appointment/{table_name}_info.csv')

    # Split the data into features (X) and target (y)
    X = df.drop(columns=[target])  # Drop the target column
    y = df[target]

    # Data preprocessing
    categorical_cols = [col for col in X.columns if X[col].dtype == object]
    numerical_cols = [col for col in X.columns if col not in categorical_cols]
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    # Build and train the model
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                      ('regressor', LinearRegression())])

    model_pipeline.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model_pipeline.predict(X_test)

    # Calculate mean square error
    mse = mean_squared_error(y_test, y_pred)
    print('The mean square error is:', mse)

    # Save the model
    joblib.dump(model_pipeline, f'D:/ticket/appointment/{table_name}_model.pkl')


if __name__ == "__main__":
    table_name = input("Enter the table name: ")
    get_table_info(table_name)

    target = input("Enter the target column: ")
    build_model(table_name, target)

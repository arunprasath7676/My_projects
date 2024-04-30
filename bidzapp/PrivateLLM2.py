import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import joblib
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sqlalchemy import create_engine
from math import sqrt

# Define database connection parameters
db_user = "mediyoga"
db_password = "mediyoga2023"
db_host = "3.111.133.67"
db_name = "Medi_Yoga"

# Define the MySQL URI
mysql_uri = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"

# Create an SQLAlchemy engine
engine = create_engine(mysql_uri)


def get_table_columns(table_name):
    # Execute SQL query to get column names
    query = f"""
        SELECT COLUMN_NAME
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{db_name}'
        AND TABLE_NAME = '{table_name}'
    """
    # Execute the query and return the result as a DataFrame
    columns_df = pd.read_sql_query(query, con=engine)
    return columns_df


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
     # Read the data from the exported CSV file
    df = pd.read_csv(f'D:/ticket/appointment/{table_name}_info.csv')

     # Split the data into features (X) and target (y)
    X = df.drop(columns=[target])  # Drop the target column
    y = df[target]

     # Preprocess numerical and categorical columns separately
    numeric_features = X.select_dtypes(include=['int', 'float']).columns
    categorical_features = X.select_dtypes(include=['object']).columns

    # Define preprocessing steps for numerical and categorical columns
    numeric_transformer = Pipeline(steps=[
         ('imputer', SimpleImputer(strategy='median')),  # Impute missing values with median
        ('scaler', StandardScaler())  # Scale numerical features
    ])

    categorical_transformer = Pipeline(steps=[
         ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values with most frequent category
         ('onehot', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
     ])

     # Combine preprocessing steps for numerical and categorical columns
    preprocessor = ColumnTransformer(
         transformers=[
             ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
         ])

     # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Create and train the model pipeline
    model_pipeline = Pipeline(steps=[
         ('preprocessor', preprocessor),
        ('regressor', LinearRegression())  # Example: Linear Regression model
    ])

    model_pipeline.fit(X_train, y_train)

   # Make predictions on the test set
    y_predictions = model_pipeline.predict(X_test)

    # Evaluate the model
    mean_squared_error_value = mean_squared_error(y_test, y_predictions)   
    root_mean_squared_error = mean_squared_error_value ** 0.5
    r_squared = model_pipeline.score(X_test, y_test)

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

    # Save the model
    joblib.dump(model_pipeline, f'D:/ticket/appointment/{table_name}_model.pkl')

    def train_and_evaluate_regressor(regressor, X_train, y_train, X_test, y_test):
        regressor.fit(X_train, y_train)
        y_predictions = regressor.predict(X_test)
        score = regressor.score(X_test, y_test)
        rmse = sqrt(mean_squared_error(y_test, y_predictions))
        return score, rmse


    def linear():
        regressor = LinearRegression()
        return regressor


    def ridge():
        regressor = Ridge(alpha=.3)
        return regressor


    def lasso():
        regressor = Lasso(alpha=0.00009)
        return regressor


    def elasticnet():
        regressor = ElasticNet(alpha=1, l1_ratio=0.5)
        return regressor


    def randomforest():
        regressor = RandomForestRegressor(n_estimators=15, min_samples_split=15, criterion='mse', max_depth=None)
        return regressor


    def perceptron():
        regressor = MLPRegressor(hidden_layer_sizes=(5000,), activation='relu', solver='adam', max_iter=1000)
        return regressor


    def decisiontree():
        regressor = DecisionTreeRegressor(min_samples_split=30, max_depth=None)
        return regressor


    def adaboost():
        regressor = AdaBoostRegressor(random_state=8, loss='exponential')
        return regressor


    def extratrees():
        regressor = ExtraTreesRegressor(n_estimators=50)
        return regressor


    def gradientboosting():
        regressor = GradientBoostingRegressor(loss='ls', n_estimators=500, min_samples_split=15)
        return regressor

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


if __name__ == "__main__":
    table_name = input("Enter the table name: ")
    get_table_info(table_name)

    #features = input("Enter the features of column: ")
    target = input("Enter the target column: ")
    build_model(table_name, target)

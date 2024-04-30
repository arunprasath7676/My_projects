import os
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores.mongodb_atlas import MongoDBAtlasVectorSearch
from pymongo import MongoClient
import csv  # Import the built-in csv module

# Load environment variables from .env file
load_dotenv(find_dotenv())

# MongoDB connection string
mongodb_conn_string = "mongodb://localhost:27017/tester"

# Database and collection names
db_name = os.environ.get("MONGODB_DB_NAME", "tester")
collection_name = os.environ.get("MONGODB_COLLECTION_NAME", "tez")

# Initialize MongoDB client and collection
client = MongoClient(mongodb_conn_string)
db = client[db_name]
collection = db[collection_name]

# Path to the CSV file
csv_file_path = "D:/ticket/appointment/patients_info.csv"

# Load CSV file
data = []
with open(csv_file_path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        data.append(row)

# Convert data to binary format (0s and 1s)
binary_data = []
for row in data:
    binary_row = []
    for item in row:
        # Convert each item to binary representation
        binary_item = ''.join(format(ord(char), '08b') for char in str(item))
        binary_row.append(binary_item)
    binary_data.append(binary_row)
 
# Store binary data in MongoDB
collection.delete_many({})
collection.insert_many({"data": row} for row in binary_data)

print("Data stored successfully in MongoDB.")

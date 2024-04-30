from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from gridfs import GridFS

uri = "mongodb+srv://arunprasath:7a2xQxS3TqdbF4U2@cluster0.axbmaip.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Accessing the database
db = client['RDMS_TO_VECTOR']

# Accessing GridFS for file storage
fs = GridFS(db)

# Path to your .crdownload file
crdownload_file_path = 'D:/ticket/tabledata.crdownload'

# Open the .crdownload file and read its content as binary
with open(crdownload_file_path, 'rb') as crdownload_file:
    # Read the content of the file
    file_content = crdownload_file.read()
    
    # Store the file content in GridFS
    file_id = fs.put(file_content, filename='file.crdownload')

# Print the ID of the stored file
print("File stored with ID:", file_id)

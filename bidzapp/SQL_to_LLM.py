import pandas as pd
import google.generativeai as genai

# Function to retrieve data based on user question (keyword search)
def get_data_from_csv(question, data):
    # Split the question into words
    words = question.lower().split()
    # Filter data by rows containing any of the words in column names
    filtered_data = data[data.columns[data.columns.str.lower().str.contains('|'.join(words))]]
    # If no data found, return a message
    if filtered_data.empty:
        return "No data found matching your question."
    # Otherwise, return the DataFrame as string
    return filtered_data.to_string(index=False)

# Function to get response from GPT-3
def get_gemini_response(question, prompt):
    genai.configure(api_key="AIzaSyDOqR_kxMa70f6XMkh1js_tUbmwQqD29bk")
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content([prompt, question])
    print(prompt,question,"sangu")
    return response.text

# Assuming your CSV file is named 'patients_info.csv'
data = pd.read_csv('D:/ticket/appointment/patients_info.csv')  

question = input("Input your question: ")

# Define your prompt (tailored for keyword search)
prompt = """
You are an expert in finding information from a student data CSV file!
The CSV file has columns. For example,
* Find all patients in name class. Answer: (Return the rows where firstname is "Arun")
* Find all patients with arun in their name. Answer: (Return rows where firstname contains "Arun")
* How many patients are there? (This requires additional logic - see Option 2)
* Find the expected count of patients for the next two years. Answer(Return the rows select count(*) from csv).
"""

# Handle button click
response = get_data_from_csv(question, data)
if response == "No data found matching your question.":
    # If no data found in CSV, get response from Gemini AI
    response = get_gemini_response(question, prompt)
print("The Response is:")
print(response)

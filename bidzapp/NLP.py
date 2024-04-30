import joblib
import pandas as pd
import spacy

# Load the trained model
model = joblib.load("D:/ticket/appointment/patients_model.pkl")  

# Load English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

# Define a function to preprocess user input
def preprocess_input(user_input):
    # Tokenize and lemmatize the input text
    doc = nlp(user_input.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

# Define a function to interact with the model based on user input
def interact_with_model(user_input):
    preprocessed_input = preprocess_input(user_input)

    input_data = pd.DataFrame([preprocessed_input], columns=["first_name"])  # Assuming your model takes a single text input

    # Print input data columns
    print(input_data.columns)

    # Use the model to make predictions
    prediction = model.predict(input_data)
    return prediction[0]  # Return the predicted output


# Interactive loop to take user input and interact with the model
print("Welcome! Ask a question based on the model:")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting...")
        break
    else:
        response = interact_with_model(user_input)
        print("Model:", response)

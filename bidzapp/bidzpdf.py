from langchain_core.prompts import PromptTemplate
from langchain_core.prompts.chat import ChatPromptTemplate
from openai import OpenAI
import os
import json

os.environ['OPENAI_API_KEY'] = "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"

example_prompt = PromptTemplate(
    input_variables=["input"], 
    template="Question: {input}\n"
)

# Here are your example inputs
inputs = {
    "input": """Serial No 1: Demolition of Existing Flooring tiles with necessary TOOLS and tackles etc
Serial No 2: Demolition of Existing Toilet wall cladding and ﬂoor tiles with necessary 
tools and tackles etc, 
Serial No 3: complete Demolition of Existing Kitchen Slaps and partition wall 
Serial No 4: Applying and Laying of Water prooﬁng in Toilet Area 
Serial No 5: Pest control Anti termite treatment 
Serial No 6: Removal of Debris from the Site LOAD
Serial No 7: Built in Ledge wall with plastering for WC Back WALL Cement Plastering 
on wall in the toilet area UPVC Ventilator For Toilet 
Serial No 8: Plumbing work with supply and laying of concealed inlet Upvc and out 
lets PVC drain line with necessary wall chasing etic, complete Toilet 2. Plumbing Line 
to connect the external line. Dismantling & Removing the existing plumbing lines and 
seal the unwanted lines with necessary ﬁttings. n.,

prompt_data = change the only provide a rfc8259 compliant json format for following rfq inputs: Serial No 1: Demolition of Existing Flooring tiles with necessary TOOLS and tackles 
etc
Serial No 2: Demolition of Existing Toilet wall cladding and ﬂoor tiles with necessary 
tools and tackles etc, 
Serial No 3: complete Demolition of Existing Kitchen Slaps and partition wall 
Serial No 4: Applying and Laying of Water prooﬁng in Toilet Area 
Serial No 5: Pest control Anti termite treatment 
Serial No 6: Removal of Debris from the Site LOAD
Serial No 7: Built in Ledge wall with plastering for WC Back WALL Cement Plastering 
on wall in the toilet area UPVC Ventilator For Toilet 
Serial No 8: Plumbing work with supply and laying of concealed inlet Upvc and out 
lets PVC drain line with necessary wall chasing etic, complete Toilet 2. Plumbing Line 
to connect the external line. Dismantling & Removing the existing plumbing lines and 
seal the unwanted lines with necessary ﬁttings. n. extract each task from Serial No 1: Demolition of Existing Flooring tiles with necessary TOOLS and tackles 
etc
Serial No 2: Demolition of Existing Toilet wall cladding and ﬂoor tiles with necessary 
tools and tackles etc, 
Serial No 3: complete Demolition of Existing Kitchen Slaps and partition wall 
Serial No 4: Applying and Laying of Water prooﬁng in Toilet Area 
Serial No 5: Pest control Anti termite treatment 
Serial No 6: Removal of Debris from the Site LOAD
Serial No 7: Built in Ledge wall with plastering for WC Back WALL Cement Plastering 
on wall in the toilet area UPVC Ventilator For Toilet 
Serial No 8: Plumbing work with supply and laying of concealed inlet Upvc and out 
lets PVC drain line with necessary wall chasing etic, complete Toilet 2. Plumbing Line 
to connect the external line. Dismantling & Removing the existing plumbing lines and 
seal the unwanted lines with necessary ﬁttings. n. and it should be distinctly identified and captured as a separate entity within the json object. it is mandatory to capture and separate all the tasks begins with serial no 1:, serial no 2: serial no 3: etc., within the json object. ensure that all pertinent details associated with each task are included and accurately represented. repeat the process in case this process fails in the first attempt. """
}

generated_prompt = example_prompt.template.format(**inputs)

#print(generated_prompt,"arun")

template = "You are a helpful assistant."

human_input = inputs["input"]

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_input),  
])

#print(chat_prompt,"arun")

# Format the messages
formatted_messages = chat_prompt.format_messages()

# Retrieve the human message content
generated_prompt_text = formatted_messages[1].content

# Call OpenAI API for chat completion
client = OpenAI()

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": generated_prompt_text}
    ]
)

print (response)

# response_json = response.json()

# # Print the response
# print(json.dumps(response_json, indent=4))

# directory = r'D:\ticket\bidzjson'

# # Construct the full path to the file
# file_path = os.path.join(directory, 'response.json')

# with open(file_path, 'w') as file:
#     json.dump(response_json, file, indent=4)
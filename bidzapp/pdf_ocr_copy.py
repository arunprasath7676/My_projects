import json
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import requests
import re
import json
import string
import nltk 
# nltk.download('stopwords')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('corpus')
nltk.download('wordnet')
from nltk.corpus import stopwords
# Function to interact with ChatGPT using OpenAI API
def chat_with_gpt(prompt,api_key):
    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4",  # or the specific ChatGPT model you are using
        "messages": [{"role": "system", "content": "You are a helpful assistant."},
                     {"role": "user", "content": prompt}]
    }
    #[{"role": "system", "content": "You are a helpful assistant."},
#                     {"role": "user", "content": prompt}]
    response = requests.post(url, headers=headers, json=data)
    return response.json()

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    #tokenize the text
    tokens = nltk.word_tokenize(text)
    
    return tokens
    
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens
    
def perform_lemmatization(tokens):
    lemmatizer = nltk.WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return lemmatized_tokens
    
    
def clean_text(text):
    tokens = preprocess_text(text)
    filtered_tokens = remove_stopwords(tokens)
    lemmatized_tokens = perform_lemmatization(filtered_tokens)
    clean_text = ' '.join(lemmatized_tokens)
    return clean_text

def extract_details(json_data):
    output_dict = {}

    def recursive_extract(json_obj):
        for key, value in json_obj.items():
            if isinstance(value, dict):
                recursive_extract(value)
            else:
                output_dict[key] = value

    recursive_extract(json_data)
    return output_dict

def is_valid_json(my_json_string):
    try:
        json_object = json.loads(my_json_string)
    except ValueError as e:
        return False
    return True

def concatenate_keys(json_data):
    concatenated_keys = ""
    for key in json_data.keys():
        concatenated_keys += f'{key}, '
    concatenated_keys = concatenated_keys[:-2]  
    return concatenated_keys

def flatten_json(json_obj, parent_key='', separator='_'):
    print('abc')
    items = {}
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key
            if isinstance(value, (dict, list)):
                items.update(flatten_json(value, new_key, separator))
            else:
                items[new_key] = value
    elif isinstance(json_obj, list):
        for i, element in enumerate(json_obj):
            new_key = f"{parent_key}{separator}{i}" if parent_key else str(i)
            if isinstance(element, (dict, list)):
                items.update(flatten_json(element, new_key, separator))
            else:
                items[new_key] = element
    else:
        items[parent_key] = json_obj

    return items

    

def pdf_to_ocr_text(pdf_path):
    # Open the PDF file
        
    pdf_document = fitz.open(pdf_path)
    
    # Initialize an empty string to store the OCR text
    ocr_text = ""

    # Iterate through each page in the PDF
    for page_num in range(pdf_document.page_count):
        # Get the page
        page = pdf_document[page_num]

        # Convert the PDF page to an image
        image_list = page.get_pixmap()

        # Convert the image to a PIL image
        pil_image = Image.frombytes("RGB", [image_list.width, image_list.height], image_list.samples)

        # Perform OCR on the image
        page_text = pytesseract.image_to_string(pil_image, lang='eng')

        # Append the OCR text from the current page to the overall OCR text
        ocr_text += page_text + "\n"
    
    
    # Close the PDF document
    pdf_document.close()
    ocr_text = clean_text(ocr_text)
    return ocr_text

## Replace 'your_pdf_file.pdf' with the path to your PDF file
pdf_path = '.\SampleFiles\CCTVSourcingvendorQuotations\securitysolutions\Security360solutionsRevisedquote2.pdf'
#pdf_path = 'C:\\Users\\User\\Desktop\\read\\MVM foam quote.pdf'
result = pdf_to_ocr_text(pdf_path)
print('result_start',result,'result_end')
clean_text = clean_text(result)
print("cleaned_text", clean_text ,'cleaned_end')
#pdf_path2 = 'Quotation_06.pdf' 
#result2 = pdf_to_ocr_text(pdf_path2)

gpt_prompt = f"Change the JSON format for following RFQ inputs:  descriptions  Channel 2MP Hikvision DVR 1 SATA 1 Audio88 Channel DVR  H.264 Dual-stream video compression Support both TVI and Analogue camerasFull channelHD1080P Resolution realtime recording, HDMI and VGA output at up to 1080*720P Resolution TB HDD Seagate 2yrs Surveillance Harddisk to store video footage. About 1.1 GB of data space per hour per camera is utilized at full frame rate and full resolution recording,HSN 852190009,QTY 1 PCS, UNITPRICE 9000,GGST 8%,GST 18% AMOUNT 90  and represent them in single JSON object"

print('gpt_prompt_text',gpt_prompt,'prompt_end_text')
# Get a response from ChatGPT
api_key = "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"
gpt_response = chat_with_gpt(gpt_prompt, api_key)
#print(gpt_response)
if 'choices' in gpt_response:
   input_string = gpt_response['choices'][0]['message']['content']
   
#print(input_string)

string_with_json = input_string.strip()
start_index = string_with_json.find('{')
end_index = string_with_json.rfind('}') + 1

json_content = string_with_json[start_index:end_index]
json_data = json.loads(json_content)

json_op = extract_details(json_data)
print(json_op)
#concatenated_keys = concatenate_keys(json_op)
#print('concated_text',concatenated_keys)
keys_params = {'Item & Description','SIZE','RATE','GST','CGST','SGST','Amount'}
#concatenated_keys = concatenate_keys(keys_params)

gpt_prompt2 = f"Can you please provide the make or manufacturer information for the product described in the JSON object on following information :\n{input_string}or similar specifications: \n{clean_text} and represent them in a JSON object format with the following:\n{keys_params}"
print('gpt_prompt_testing', gpt_prompt2, 'gpt_prompt_ending')
gpt_response = chat_with_gpt(gpt_prompt2, api_key)
print('get_response', gpt_response, 'get_response_end2')

if 'choices' in gpt_response:
    input_string2 = gpt_response['choices'][0]['message']['content']
    print('Extracted content:', input_string2)

    start_index = input_string2.find('{')
    end_index = input_string2.rfind('}') + 1
    json_string = input_string2[start_index:end_index]    

    json_data = json.loads(json_string)
    print('Extracted JSON data:', json_data,'extracted end')
    json_op = extract_details(json_data)
    print(json_op)
    concatenated_keys = concatenate_keys(json_op)
    print('Concatenated text:', concatenated_keys)


  json_op1 = '''
    {
        "ProposalFor": {
            "System": "2MP HD CCTV Surveillance System"
        },
        "Specifications": [
            {
                "Description": "Pie Dare 20 Meir + Paszbocy R Dame Canara Mode Ts",
                "Qty": "ana0",
                "Rate": "ronan",
                "Amount": "1"
            },
            {
                "Description": "PRAMA AHD 2MP 8 Channel DVR, Model: PT-DR1AOSG-K1",
                "Qty": 1,
                "Rate": 6700.00,
                "Amount": 6700.00
            },
            {
                "Description": "12vot DG 8 Amps Switched Mode Power Supply",
                "Qty": 1,
                "Rate": 1050.00,
                "Amount": 1050.00
            },
            {
                "Description": "2TB CCTV Surveillance Hard Disk, Make: Seagate-SkyHawk",
                "Qty": 1,
                "Rate": 4900.00,
                "Amount": 4900.00
            },
            {
                "Description": "8 DC Connector",
                "Qty": 15,
                "Rate": 40.00,
                "Amount": 600.00
            },
            {
                "Description": "Camera installation PVC Box",
                "Qty": 5,
                "Rate": 75.00,
                "Amount": 275.00
            },
            {
                "Description": "lnstallation, Testing & Commissioning Charges",
                "Qty": 1,
                "Rate": 3000.00,
                "Amount": 3000.00
            }
        ],
        "TotalCost": {
            "WithoutTax": 29275.00,
            "With18PercentGST": 34548.50
        }
    }
    '''

new_data = flatten_json(json_op1)
print('abc', new_data)



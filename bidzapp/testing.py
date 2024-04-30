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



json_op = '''
{
  "RFQ_Inputs": {
    "Processor": "Intel Core i5/i7 (11th Gen)",
    "RAM": "16 GB LPDDR4x",
    "Storage": "512 GB NVMe SSD",
    "Display": "14-inch Full HD (1920 x 1080) with HDR support",
    "Graphics": "Integrated Intel Iris Xe Graphics",
    "OS": "Windows 10 Pro or Windows 11 Pro",
    "Ports": "2 x Thunderbolt 4 (USB-C), 2 x USB-A, 3.5mm headphone/microphone combo jack, microSD card reader",
    "Wireless": "Wi-Fi 6, Bluetooth 5.0",
    "Battery": "4-cell 57WHr integrated battery",
    "Keyboard and Touchpad": "Backlit chiclet keyboard, Precision touchpad",
    "Audio": "3.5mm headphone/microphone combo jack",
    "Webcam": "HD 720p Webcam"
  },
  "Quotation": {
    "Offer1": {
      "Processor": "Intel Core i3 12100T",
      "RAM": "8 GB DDR4",
      "Storage": "512 GB SSD",
      "Quantity": "11000",
      "Price per unit": "35673",
      "Model": "H610 N4690G"
    },
    "Offer2": {
      "Processor": "Intel Core i3 12100T",
      "RAM": "8 GB DDR4",
      "Storage": "512 GB SSD",
      "Quantity": "2600",
      "Price per unit": "32267",
      "Model": "H610 N4690G"
    },
    "Offer3": {
      "Processor": "Intel Corei3 1215U",
      "RAM": "8 GB DDR4",
      "Storage": "512GB SSD",
      "Display": "14 HD 1366768",
      "Quantity": "200",
      "Price per unit": "35648",
      "Model": "TMP214 54 12th Gen"
    }
  }
}
'''
json_data = json.loads(json_op)
new_data = flatten_json(json_data)
print('abc', new_data)
new_data_keys = list(new_data.keys())
print('json_keys:',new_data_keys)

data_list = [{"file1": "qw"}, {"file2": "qq"}]



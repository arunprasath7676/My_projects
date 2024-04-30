import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_url, cached_download
import PIL
# import onnx
import onnxruntime
import os
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision.models
import torch.onnx
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import albumentations
import albumentations.pytorch.transforms
import cv2
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from gradio import Interface, inputs, outputs
import onnx
import onnxruntime
import coremltools
import mlblocks
import megengine.functional as mb

config_file_url = hf_hub_url("Jacopo/ToonClip", filename="model.onnx")
model_file = cached_download(config_file_url)

onnx_model = onnx.load(model_file)
onnx.checker.check_model(onnx_model)

opts = onnxruntime.SessionOptions()
opts.intra_op_num_threads = 16
ort_session = onnxruntime.InferenceSession(model_file, sess_options=opts)

input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

def normalize(x, mean=(0., 0., 0.), std=(1.0, 1.0, 1.0)):
    # x = (x - mean) / std
    x = np.asarray(x, dtype=np.float32)
    if len(x.shape) == 4:
        for dim in range(3):
            x[:, dim, :, :] = (x[:, dim, :, :] - mean[dim]) / std[dim]
    if len(x.shape) == 3:
        for dim in range(3):
            x[dim, :, :] = (x[dim, :, :] - mean[dim]) / std[dim]

    return x 

def denormalize(x, mean=(0., 0., 0.), std=(1.0, 1.0, 1.0)):
    # x = (x * std) + mean
    x = np.asarray(x, dtype=np.float32)
    if len(x.shape) == 4:
        for dim in range(3):
            x[:, dim, :, :] = (x[:, dim, :, :] * std[dim]) + mean[dim]
    if len(x.shape) == 3:
        for dim in range(3):
            x[dim, :, :] = (x[dim, :, :] * std[dim]) + mean[dim]

    return x 

def nogan(input_img):
    i = np.asarray(input_img)
    i = i.astype("float32")
    i = np.transpose(i, (2, 0, 1))
    i = np.expand_dims(i, 0)
    i = i / 255.0
    i = normalize(i, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    ort_outs = ort_session.run([output_name], {input_name: i})
    output = ort_outs
    output = output[0][0]

    output = denormalize(output, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    output = output * 255.0
    output = output.astype('uint8')
    output = np.transpose(output, (1, 2, 0))
    output_image = PIL.Image.fromarray(output, 'RGB')

    return output_image

title = "ToonClip Comics Hero Demo"
description = "..."
article = "..."

iface = gr.Interface(
    nogan, 
    gr.inputs.Image(type="pil", shape=(1024, 1024)),
    gr.outputs.Image(type="pil"),
    title=title,
    description=description,
    article=article,
    enable_queue=True,
    live=True)

iface.launch()


# Define constants or parameters
INPUT_IMAGE_PATH = "D:/ticket/huggingface/Sampleimage.jpeg"
OUTPUT_IMAGE_PATH = "D:/ticket/huggingface/comic_output.jpg"

# Load the input image
input_image = cv2.imread(INPUT_IMAGE_PATH)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Preprocess the input image
input_image = cv2.resize(input_image, (512, 512))  # Resize to match the model input size
input_image = input_image / 255.0  # Normalize pixel values

# Convert the input image to torch tensor
input_tensor = torch.tensor(input_image, dtype=torch.float32)
input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

# Load the trained model
model = ImageToImageModel.load_from_checkpoint("path_to_your_checkpoint_file.ckpt")

# Perform the conversion
model.eval()
with torch.no_grad():
    output_tensor = model(input_tensor)

# Post-process the output image
output_image = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
output_image = (output_image * 255).astype(np.uint8)
output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

# Save the output image to the specified folder path
cv2.imwrite(OUTPUT_IMAGE_PATH, output_image)

print("Comic image saved successfully at:", OUTPUT_IMAGE_PATH)

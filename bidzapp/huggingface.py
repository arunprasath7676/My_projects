import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_url, cached_download
import PIL
import onnx
import onnxruntime

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
examples=[['i01.jpeg'], ['i02.jpeg'], ['i03.jpeg'], ['i04.jpeg'], ['i05.jpeg'], ['i06.jpeg'], ['i07.jpeg'], ['i08.jpeg'], ['i09.jpeg'], ['i10.jpeg']]

iface = gr.Interface(
    nogan, 
    gr.inputs.Image(type="pil", shape=(1024, 1024)),
    gr.outputs.Image(type="pil"),
    title=title,
    description=description,
    article=article,
    examples=examples,
    enable_queue=True,
    live=True)

iface.launch()
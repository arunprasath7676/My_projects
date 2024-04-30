import os
import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torchvision.models
import torch.optim
import torch.optim.lr_scheduler
import torch.utils.data
import albumentations
import albumentations.pytorch.transforms
import cv2
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import gradio as gr
import torchvision.models as models



# Define constants or parameters
INPUT_IMAGE_PATH = "D:/ticket/huggingface/Sampleimage.jpg"
OUTPUT_IMAGE_PATH = "D:/ticket/huggingface/comic_output.jpg"

# Define the ImageToImageModel class
class ImageToImageModel(pl.LightningModule):
    def __init__(self, net, trainset, valset, batch_size=32, learning_rate=0.001, content_loss_weight=1.0, 
                 total_style_loss_weight=1.0, style_loss_weights=[1, 1, 1, 1], style_layers_weight_power=1):
        super(ImageToImageModel, self).__init__()
        self.net = net
        self.trainset = trainset
        self.valset = valset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.content_loss_weight = content_loss_weight
        self.total_style_loss_weight = total_style_loss_weight
        self.style_loss_weights = style_loss_weights
        self.style_layers_weight_power = style_layers_weight_power
       

    def forward(self, x):
        return self.net(x)
       

    def training_step(self, batch, batch_nb):
        image, gtruth = batch
        prediction = self.forward(image)
        content_loss = F.l1_loss(prediction, gtruth)
        style_loss = vgg_loss(prediction, gtruth, 
                             feature_layers_weight=self.style_loss_weights, 
                             style_layers_weight=[k**self.style_layers_weight_power for k in self.style_loss_weights])
        style_loss = style_loss * self.total_style_loss_weight
        # reg_loss TODO
        total_loss = (self.content_loss_weight * content_loss) + style_loss
        self.log("content_loss", content_loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("style_loss", style_loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("total_loss", total_loss.detach(), on_step=False, on_epoch=True, prog_bar=False)
        return {'loss' : total_loss}

    def validation_step(self, batch, batch_idx):
        image, gtruth = batch
        prediction = self.forward(image)
        content_loss = F.l1_loss(prediction, gtruth)
        style_loss = vgg_loss(prediction, gtruth, 
                             feature_layers_weight=self.style_loss_weights, 
                             style_layers_weight=[k**self.style_layers_weight_power for k in self.style_loss_weights])
        style_loss = style_loss * self.total_style_loss_weight
        # reg_loss TODO
        total_loss = (self.content_loss_weight * content_loss) + style_loss
        self.log("val_content_loss", content_loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_style_loss", style_loss.detach(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_total_loss", total_loss.detach(), on_step=False, on_epoch=True, prog_bar=False)
        return {'val_loss' : total_loss}

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr = self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max = 10)
        return [opt], [sch]

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size = self.batch_size, shuffle = True, num_workers = 4)

    def val_dataloader(self):
        return DataLoader(self.valset, batch_size = self.batch_size, shuffle = False, num_workers = 4)


# Load the trained model
# model = ImageToImageModel.load_from_checkpoint("path_to_your_checkpoint_file.ckpt")

# Load a pre-trained ResNet model
model = models.resnet18(pretrained=True)

# Load the input image
input_image = cv2.imread(INPUT_IMAGE_PATH)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Preprocess the input image
input_image = cv2.resize(input_image, (512, 512))  # Resize to match the model input size
input_image = input_image / 255.0  # Normalize pixel values

# Convert the input image to torch tensor
input_tensor = torch.tensor(input_image, dtype=torch.float32)
input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

model.eval()
with torch.no_grad():
    output_tensor = model(input_tensor)

# Post-process the output image
output_tensor = output_tensor.squeeze(0) 
output_image = None  # Default value

if len(output_tensor.shape) == 3:
    output_image = output_tensor.permute(1, 2, 0).cpu().numpy()
else:
    print("Error: Unexpected tensor shape. Unable to generate output image.")

if output_image is not None:
    output_image = (output_image * 255).astype(np.uint8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

    # Save the output image to the specified folder path
    cv2.imwrite(OUTPUT_IMAGE_PATH, output_image)
    print("Comic image saved successfully at:", OUTPUT_IMAGE_PATH)
else:
    print("Error: Output image not generated. Unable to save.")

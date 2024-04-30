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

class ImageToImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_path, output_path, size=512, transform=None):
        self.size = size
        self.input_path = input_path
        self.output_path = output_path
        self.transform = transform
        
        self.input_list = self.get_filenames(self.input_path)
        self.output_list = self.get_filenames(self.output_path)
        
    def __len__(self):
        return len(self.input_list)
    
    def __getitem__(self, idx):
        input_img = cv2.imread(self.input_list[idx])
        input_img = cv2.resize(input_img, (self.size, self.size))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        output_img = cv2.imread(self.output_list[idx])
        output_img = cv2.resize(output_img, (self.size, self.size))
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            augmented = self.transform(image=input_img)
            input_img = augmented['image']
            augmented = albumentations.ReplayCompose.replay(augmented['replay'], image=output_img)
            output_img = augmented['image']
        
        return input_img, output_img
    
    def get_filenames(self, path):
        files_list = []
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
            
        return files_list

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize

    def forward(self, input, target, feature_layers_weight=[1, 1, 1, 1], style_layers_weight=[1, 1, 1, 1]):
       if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if feature_layers_weight[i] > 0:
                feature_weight = feature_layers_weight[i]
                loss += feature_weight * torch.nn.functional.l1_loss(x, y)
            if style_layers_weight[i] > 0:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                style_weight = style_layers_weight[i]
                loss += style_weight * torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss
		
		

model = SemanticSegmentation(
    head = HEAD,
    backbone = BACKBONE,
    pretrained = PRETRAINED, 
    num_classes = 3)

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


    i2imodel = ImageToImageModel(model, 
                             train_set, 
                             val_set,
                             batch_size = BATCH_SIZE,
                             learning_rate = LEARNING_RATE,
                             content_loss_weight = CONTENT_LOSS_WEIGHT,
                             total_style_loss_weight = TOTAL_STYLE_LOSS_WEIGHT,
                             style_loss_weights = STYLE_LOSS_WEIGHTS,
                             style_layers_weight_power = STYLE_LAYERS_WEIGHT_POWER
                            )

    trainer = pl.Trainer(gpus = NUM_GPUS, 
                     max_epochs = EPOCHS,
                     log_every_n_steps = 25,
                     check_val_every_n_epoch=1,
                     enable_checkpointing=False
                    )

trainer.fit(i2imodel)

def main():
    # Define constants or parameters
    # Load or define models, datasets, loss functions, etc.
    # Train the model
    # Convert the model to ONNX
    # Deploy the model using Gradio
    # Convert the model to CoreML

if __name__ == "__main__":
    main()

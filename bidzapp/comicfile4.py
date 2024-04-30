import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np

# Define the path to the input image and the output image
INPUT_IMAGE_PATH = "D:/ticket/huggingface/Sampleimage.jpg"
OUTPUT_IMAGE_PATH = "D:/ticket/huggingface/comic_output.jpg"

vgg = models.vgg19(pretrained=True).features

# Define a function to perform style transfer
def style_transfer(content_img, num_steps=300, style_weight=1000000, content_weight=1):
    # Set up image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    # Preprocess the content image
    content = transform(content_img).unsqueeze(0)
    
    # Clone the content image
    target = content.clone().requires_grad_(True)
    
    # Set up content features
    content_features = get_features(content, vgg)
    
    # Calculate the Gram matrices for the content features
    content_grams = {layer: gram_matrix(content_features[layer]) for layer in content_features}
    
    # Set up optimizer
    optimizer = torch.optim.Adam([target], lr=0.003)
    
    for i in range(1, num_steps+1):
        # Get the features of the target image
        target_features = get_features(target, vgg)
        
        # Calculate the content loss
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        
        # Calculate the style loss
        style_loss = 0
        for layer in content_grams:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape
            content_gram = content_grams[layer]
            layer_style_loss = content_grams[layer] * torch.mean((target_gram - content_gram)**2)
            style_loss += layer_style_loss / (d * h * w)
        
        # Calculate the total loss
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        # Update the target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Output intermediate images
        if i % 50 == 0:
            print(f'Step [{i}/{num_steps}], Total Loss: {total_loss.item()}')
            output_img = target.squeeze(0).permute(1, 2, 0).detach().numpy()
            output_img = np.clip(output_img, 0, 1)
            cv2.imwrite(f'output_{i}.jpg', output_img * 255)
    
    return target.squeeze(0).permute(1, 2, 0).detach().numpy()

# Define helper functions
def get_features(image, model):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if isinstance(layer, nn.Conv2d):
            features[name] = x
            print(f"Added features from layer: {name}")
    return features


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Load the input image
content_img = cv2.imread(INPUT_IMAGE_PATH)
content_img = cv2.cvtColor(content_img, cv2.COLOR_BGR2RGB)

# Perform style transfer
output_img = style_transfer(content_img)

# Save the output image
cv2.imwrite(OUTPUT_IMAGE_PATH, cv2.cvtColor(output_img * 255, cv2.COLOR_RGB2BGR))
print("Image saved successfully at:", OUTPUT_IMAGE_PATH)

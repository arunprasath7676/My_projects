import os
import numpy as np
import cv2
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Flatten, Dense, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

# Function to load and preprocess images
def load_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))  # Resize image to 64x64 pixels
            img = img / 255.0  # Normalize pixel values to [0, 1]
            images.append(img)
    return np.array(images)

# Load and preprocess cartoon face pairs
source_cartoon_faces = load_images("C:/Users/User/Downloads/cartoon_img1.jpg")
target_cartoon_faces = load_images("C:/Users/User/Downloads/cartoon_img2.jpg")

# Define the generator network
def build_generator():
    input_layer = Input(shape=(64, 64, 3))
    
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Flatten()(x)
    x = Dense(128)(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Dense(64 * 64 * 3, activation='sigmoid')(x)
    generated_image = Reshape((64, 64, 3))(x)
    
    model = Model(input_layer, generated_image)
    return model

# Define the discriminator network
def build_discriminator():
    input_layer = Input(shape=(64, 64, 3))
    
    x = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(input_layer)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    
    x = Flatten()(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(input_layer, x)
    return model

# Compile the discriminator (only for training purposes)
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# Define the combined generator and discriminator model
generator = build_generator()
z = Input(shape=(100,))
generated_image = generator(z)
discriminator.trainable = False
validity = discriminator(generated_image)
combined_model = Model(z, validity)
combined_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0002, beta_1=0.5))

# Training loop
batch_size = 32
epochs = 10000

for epoch in range(epochs):
    # Select a random batch of source cartoon faces
    idx = np.random.randint(0, source_cartoon_faces.shape[0], batch_size)
    source_cartoon_batch = source_cartoon_faces[idx]
    
    # Generate a batch of fake cartoon faces
    noise = np.random.normal(0, 1, (batch_size, 100))
    generated_cartoon_batch = generator.predict(noise)
    
    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(source_cartoon_batch, np.ones((batch_size, 1)))
    d_loss_fake = discriminator.train_on_batch(generated_cartoon_batch, np.zeros((batch_size, 1)))
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    
    # Train the generator
    noise = np.random.normal(0, 1, (batch_size, 100))
    g_loss = combined_model.train_on_batch(noise, np.ones((batch_size, 1)))
    
    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, D Loss: {d_loss}, G Loss: {g_loss}")

# Save the generator model
generator.save("cartoon_face_swapper_generator.h5")

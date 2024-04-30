import tensorflow as tf
import matplotlib.pyplot as plt

def build_faceswap_model(image_height, image_width):
    # Define the encoder and decoder architecture for the face swap model.
    inputs = tf.keras.layers.Input(shape=(image_height, image_width, 3))  # Assuming RGB images
    # Encoder
    encoded = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    encoded = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    encoded = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
    encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
    # Decoder
    decoded = tf.keras.layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same')(encoded)
    decoded = tf.keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = tf.keras.layers.Conv2DTranspose(128, (3, 3), activation='relu', padding='same')(decoded)
    decoded = tf.keras.layers.UpSampling2D((2, 2))(decoded)
    decoded = tf.keras.layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same')(decoded)
    decoded = tf.keras.layers.UpSampling2D((2, 2))(decoded)
    outputs = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoded)  # Output RGB image

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Load pre-trained face swap model (if applicable)
image_height = 128  # Update with actual image height
image_width = 128  # Update with actual image width
model = build_faceswap_model(image_height, image_width)
# model.load_weights("path_to_model_weights.h5")  # Load pre-trained weights

# Load example cartoon images for face swapping
cartoon_img1_path = "C:/Users/User/Downloads/cartoon_img1.jpg"
cartoon_img2_path = "C:/Users/User/Downloads/cartoon_img2.jpg"

cartoon_img1 = tf.keras.preprocessing.image.load_img(cartoon_img1_path, target_size=(image_height, image_width))
cartoon_img2 = tf.keras.preprocessing.image.load_img(cartoon_img2_path, target_size=(image_height, image_width))

cartoon_img1_array = tf.keras.preprocessing.image.img_to_array(cartoon_img1) / 255.0
cartoon_img2_array = tf.keras.preprocessing.image.img_to_array(cartoon_img2) / 255.0

cartoon_img1_array = tf.expand_dims(cartoon_img1_array, axis=0)
cartoon_img2_array = tf.expand_dims(cartoon_img2_array, axis=0)

# Perform face swapping
swapped_image = model(cartoon_img2_array)[0]  # Swapping cartoon_img2 onto cartoon_img1

# Plot original and swapped images
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.title("Cartoon Image 1")
plt.imshow(cartoon_img1)

plt.subplot(1, 3, 2)
plt.title("Cartoon Image 2")
plt.imshow(cartoon_img2)

plt.subplot(1, 3, 3)
plt.title("Swapped Image")
plt.imshow(swapped_image)
plt.show()

# Save the swapped image
swapped_image_path = "C:/Users/User/Downloads/swapped_cartoon_image.jpg"
tf.keras.preprocessing.image.save_img(swapped_image_path, swapped_image)

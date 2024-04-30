import os
import numpy as np
import cv2
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import dlib  # For facial landmarks detection
from imutils import face_utils

import os

print(os.path.exists("/home/Downloads/bidz/bidzapp/shape_predictor_68_face_landmarks.dat"),"test")


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

# Load facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("C:/bidz/model/shape_predictor_68_face_landmarks.dat")  # Download the model

# Function to detect facial landmarks
def detect_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    if len(rects) > 0:
        shape = predictor(gray, rects[0])
        shape = face_utils.shape_to_np(shape)
        return shape
    else:
        return None

# Function to swap faces
def face_swap(source_face, target_face):
    # Detect facial landmarks
    source_landmarks = detect_landmarks(source_face)
    target_landmarks = detect_landmarks(target_face)

    if source_landmarks is None or target_landmarks is None:
        return None

    # Define source and target points for warping
    source_points = np.array(source_landmarks, dtype=np.float32)
    target_points = np.array(target_landmarks, dtype=np.float32)

    # Compute affine transformation
    M = cv2.estimateAffinePartial2D(source_points, target_points)[0]
    
    # Warp source face to align with target face
    warped_source_face = cv2.warpAffine(source_face, M, (target_face.shape[1], target_face.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    return warped_source_face

# After the training loop:
# Choose a source cartoon face and a target cartoon face
source_index = np.random.randint(0, len(source_cartoon_faces))
target_index = np.random.randint(0, len(target_cartoon_faces))

source_face = source_cartoon_faces[source_index]
target_face = target_cartoon_faces[target_index]

# Perform face swapping
swapped_face = face_swap(source_face, target_face)

# Display the result
plt.figure(figsize=(8, 4))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(source_face, cv2.COLOR_BGR2RGB))
plt.title('Source Face')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(target_face, cv2.COLOR_BGR2RGB))
plt.title('Target Face')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(swapped_face, cv2.COLOR_BGR2RGB))
plt.title('Swapped Face')
plt.axis('off')

plt.show()

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from tensorflow.keras.applications import VGG16

model = VGG16(weights="static/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False)
# Load dataset
def load_images_from_folder(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        class_path = os.path.join(folder, label)
        if os.path.isdir(class_path):
            for file in os.listdir(class_path):
                img_path = os.path.join(class_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                img = cv2.resize(img, (128, 128))  # Resize for CNN
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

# Extract GLCM features
def extract_glcm_features(image):
    distances = [1, 2, 3, 4]  # Pixel distances
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Angles
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)

    features = []
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    for prop in properties:
        values = graycoprops(glcm, prop)
        features.extend(values.flatten())
    
    return features

# Load pretrained CNN (VGG16)
def extract_cnn_features(model, image):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
    image = cv2.resize(image, (224, 224))  # Resize for VGG16
    image = img_to_array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Expand dimensions
    features = model.predict(image)
    return features.flatten()

# Load images
dataset_path = "static/data"  # Change this to your dataset path
images, labels = load_images_from_folder(dataset_path)

# Encode labels
label_mapping = {label: idx for idx, label in enumerate(np.unique(labels))}
labels = np.array([label_mapping[label] for label in labels])

# Extract GLCM features
glcm_features = np.array([extract_glcm_features(img) for img in images])

# Load Pretrained VGG16 Model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
cnn_model = Model(inputs=base_model.input, outputs=base_model.output)

# Extract CNN features
cnn_features = np.array([extract_cnn_features(cnn_model, img) for img in images])

# Combine GLCM and CNN features
features = np.hstack((glcm_features, cnn_features))

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train classifier (SVM)
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Predict and evaluate
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy:.2f}")

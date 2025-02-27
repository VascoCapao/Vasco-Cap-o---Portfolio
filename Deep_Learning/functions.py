# --------------------------------------
# All imports for all notebooks
# --------------------------------------

import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import cv2

from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    Resizing, GlobalAveragePooling2D
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2, NASNetMobile, InceptionV3

import keras_tuner
from kerastuner.tuners import Hyperband


# --------------------------------------
# Utility Functions
# --------------------------------------

def extract_info(path):
    """
    Extracts information about the image based on its file path to fill missing values
    
    Parameters:
        path (str): The file path of the image.

    Returns:
        tuple: Contains information (benign/malignant, cancer type, magnification) or NaN values if extraction fails.
    """
    parts = path.split('/')
    if len(parts) >= 6:
        benign_malignant = parts[3].capitalize()
        cancer_type = parts[5].replace("_", " ").title()
        magnification = parts[7]
        return benign_malignant, cancer_type, magnification
    return np.nan, np.nan, np.nan

def load_and_preprocess_image(image_path, target_size=(150, 150)):
    """
    Loads and preprocesses an image for model input.
    
    Parameters:
        image_path (str): The file path of the image.
        target_size (tuple): The target size to resize the image.

    Returns:
        np.array: The preprocessed image.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    img = cv2.resize(img, target_size)
    img = img / 255.0
    return img

def load_images(image_paths, target_size=(150, 150)):
    """
    Loads and preprocesses multiple images.
    
    Parameters:
        image_paths (list): List of image file paths.
        target_size (tuple): The target size to resize the images.

    Returns:
        np.array: Array of preprocessed images.
    """
    return np.array([load_and_preprocess_image(img_path, target_size) for img_path in image_paths])

# --------------------------------------
# Plotting Functions
# --------------------------------------

def plot_training_history(history):
    """
    Plots the training and validation accuracy/loss over epochs.
    
    Parameters:
        history: Training history object from Keras.
    """
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm, class_names, title):
    """
    Plots a confusion matrix.
    
    Parameters:
        cm (np.array): Confusion matrix.
        class_names (list): Names of the classes.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Reds', values_format='d', ax=plt.gca())
    plt.title(title)
    plt.show()

def plot_diferent_images(X_train_images, img_indx, transformation, factor=None):
    """
    Plot an original image alongside its transformed version.

    Parameters:
    X_train_images (numpy array): Array of training images.
    img_indx (int): Index of the image to transform and plot.
    transformation (function): Function to apply a transformation to the image.
    factor (float, optional): Factor to control the intensity of the transformation.

    Returns:
    None: Displays a side-by-side plot of the original and transformed images.
    """
    # Select the original image using the provided index
    image = X_train_images[img_indx]

    # Apply the transformation with or without the factor
    if factor is not None:
        image_transformed = transformation(image, factor)
    else:
        image_transformed = transformation(image)

    # Create a side-by-side plot of the original and transformed images
    plt.figure(figsize=(10, 5))

    # Plot the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    # Plot the transformed image
    plt.subplot(1, 2, 2)
    plt.imshow(image_transformed)
    plt.title('Transformed Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# --------------------------------------
# Transformations
# --------------------------------------

def adjust_brightness(images, factor=0.1):
    """
    Adjusts image brightness by adding a constant value to all pixels.
    
    Parameters:
        images (np.array): Input images.
        factor (float): Brightness adjustment factor.

    Returns:
        np.array: Brightness-adjusted images.
    """
    return np.clip(images + factor, 0, 1)

def adjust_contrast(images, factor=0.4):
    """
    Adjust the contrast of images by scaling pixel intensity values.

    Parameters:
    images (numpy array): Input images.
    factor (float): Contrast adjustment factor.

    Returns:
        np.array: Contrast-adjusted images.
    """
    # Compute the mean pixel intensity for each image
    mean = np.mean(images, axis=(1, 2), keepdims=True)
    # Adjust contrast by scaling pixel values relative to the mean
    return np.clip((images - mean) * factor + mean, 0, 1)


def adjust_saturation(images, factor=1.5):
    """
    Adjust the saturation of RGB images using the HSV color space.

    Parameters:
    images (numpy array): Input images.
                          
    factor (float): Saturation adjustment factor. 

    Returns:
        np.array: Images with adjusted saturation, in RGB format.
    """
    # Ensure the input images are in the correct data type
    if images.dtype == np.float64:
        images = images.astype(np.float32)

    # Convert images from RGB to HSV color space
    images_hsv = cv2.cvtColor(images, cv2.COLOR_RGB2HSV)
    # Adjust the saturation channel
    images_hsv[..., 1] = np.clip(images_hsv[..., 1] * factor, 0, 1)
    # Convert images back to RGB color space
    return cv2.cvtColor(images_hsv, cv2.COLOR_HSV2RGB)


def add_noise(images, std=0.2):
    """
    Add random Gaussian noise to images.

    Parameters:
    images (numpy array): Input images.
    std (float): Standard deviation of the Gaussian noise.

    Returns:
    numpy array: Images with added noise, clipped between 0 and 1.

    """
    # Generate random noise with a mean of 0 and specified standard deviation
    noise = np.random.normal(0, std, images.shape)
    # Add noise to the images and clip values to valid range
    return np.clip(images + noise, 0, 1)

# --------------------------------------
# Model Creation
# --------------------------------------

def create_pretrained_model(base_model_name, input_shape=(150, 150, 3), num_classes=1):
    """
    Creates a model with a pre-trained base and custom classification head.
    
    Parameters:
        base_model_name (str): The name of the pre-trained base model to use.
        input_shape (tuple): Input shape for the model.
        num_classes (int): Number of output classes.

    Returns:
        keras.models.Sequential: The compiled model.
    """
    model = Sequential()
    model.add(Resizing(224, 224, input_shape=input_shape))

    if base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    elif base_model_name == 'NasNetMobile':
        base_model = NASNetMobile(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    elif base_model_name == 'InceptionV3':
        base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    else:
        raise ValueError("Unsupported model name. Choose from 'MobileNetV2', 'NasNetMobile', 'InceptionV3'.")

    base_model.trainable = False
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax'))
    
    model.compile(
        optimizer='rmsprop',
        loss='binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_with_transformation(X_train, y_train, X_val, y_val, transformation_name, transformation_label, class_weights=None, **kwargs):
    """
    Trains a model with transformations applied to the dataset.

    Parameters:
        X_train (np.array): Training data.
        y_train (np.array): Training labels.
        X_val (np.array): Validation data.
        y_val (np.array): Validation labels.
        transformation_name (function): Transformation function to apply.
        transformation_label (str): Label for the transformation.
        class_weights (dict, optional): Class weights for training.
        **kwargs: Additional arguments for the transformation function.

    Returns:
        history: Training history object from Keras.
    """
    # Apply transformation
    X_train_transformed = np.array([transformation_name(img, **kwargs) for img in X_train])
    X_val_transformed = np.array([transformation_name(img, **kwargs) for img in X_val])

    early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=4,
    restore_best_weights=True)
    
    # Model Architecture
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(8, activation='softmax'))

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator()  

    train_generator = train_datagen.flow(
        X_train_transformed, y_train,
        batch_size=40,
        shuffle=True
    )

    val_generator = val_datagen.flow(
        X_val_transformed, y_val,
        batch_size=40
    )
    

    # Train the model
    history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    steps_per_epoch=len(X_train) // 40,
    validation_steps=len(X_val) // 40,
    callbacks=[early_stopping],
    class_weight=class_weights)

    model_filename = f"saved_models/model_{transformation_label}.h5"
    model.save(model_filename)
    print(f"Model saved as {model_filename}")
    print(f"Class Weights used: {class_weights}")

    return history

# Function to dynamically build the model
def build_model(hp):
    """
    Builds a convolutional neural network (CNN) model dynamically using Hyperparameter tuning.

    Parameters:
        hp (HyperParameters): An instance of HyperParameters from Keras Tuner, used to specify the range of hyperparameters.

    Returns:
        Sequential: A compiled Keras Sequential model ready for training.
    
    Hyperparameters:
        - num_conv_layers (int): Number of convolutional layers (range: 1 to 4).
        - filters_{i} (int): Number of filters in each convolutional layer (range: 32 to 128, step: 32).
        - kernel_size_{i} (int): Kernel size for each convolutional layer (choices: 3, 5).
        - dense_units (int): Number of units in the dense layer (range: 32 to 256, step: 32).
        - learning_rate (float): Learning rate for the Adam optimizer (choices: 1e-2, 1e-3, 1e-4).

    Notes:
        - The model is designed for image input shapes of (150, 150, 3).
        - The output layer has 8 units with a softmax activation for multi-class classification.
    """

    model = Sequential()

    # Tunable number of convolutional layers
    for i in range(hp.Int('num_conv_layers', 1, 4)): 
        if i == 0:
            model.add(Conv2D(
                filters=hp.Int(f'filters_{i}', min_value=32, max_value=128, step=32),
                kernel_size=hp.Choice(f'kernel_size_{i}', values=[3, 5]),
                activation='relu',
                input_shape=(150, 150, 3) 
            ))
        else:
            model.add(Conv2D(
                filters=hp.Int(f'filters_{i}', min_value=32, max_value=128, step=32),
                kernel_size=hp.Choice(f'kernel_size_{i}', values=[3, 5]),
                activation='relu'
            ))

        # Add MaxPooling
        model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten and add dense layers
    model.add(Flatten())
    model.add(Dense(units=hp.Int('dense_units', min_value=32, max_value=256, step=32),
        activation='relu'
    ))
    model.add(Dense(8, activation='softmax'))  

    # Compile the model with a tunable learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
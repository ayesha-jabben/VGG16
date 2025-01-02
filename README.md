# VGG16 Image Classification

This repository contains code and resources for training and evaluating image classification models using TensorFlow/Keras. The project demonstrates custom CNN architecture and transfer learning using VGG16, a pre-trained model. It is designed to work with an (cat and dog)image dataset stored on Google Drive.

## Features

- **Google Drive Integration**: Easily access and organize your datasets stored on Google Drive.
- **Dataset Preparation**:
  - Loads images from directories.
  - Splits the dataset into training, validation, and test sets.
  - Visualizes image samples and their corresponding labels.
- **Data Augmentation**:
  - Includes random flipping, rotation, and zoom to enhance model generalization.
  - Resizing and rescaling of images to match input requirements.
- **Model Training**:
  - Custom CNN architecture defined using TensorFlow/Keras layers.
  - Transfer learning with VGG16 for leveraging pre-trained weights.
  - Evaluation on validation and test datasets.
- **Prediction**: Supports uploading new images for prediction and visualizes results.

## Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/vgg16-image-classification.git
cd vgg16-image-classification
```

### 2. Prepare Dataset

Organize your dataset into subdirectories for each class, e.g.:

```
/path/to/dataset/
  cats/
    cat1.jpg
    cat2.jpg
  dogs/
    dog1.jpg
    dog2.jpg
```

Upload the dataset to your Google Drive.

### 3. Run the Notebook

Open `VGG16.ipynb` in Google Colab and execute the cells sequentially. Update the dataset path in the notebook to match your Google Drive directory.

### 4. Train the Model

Train the model using either the custom CNN or the VGG16 transfer learning approach. Monitor the training process through loss and accuracy metrics.

### 5. Predict

Upload an image using the provided functionality in the notebook. The model will classify the image and display the result.

## Results

The notebook generates visualizations for:
- Training and validation loss/accuracy.
- Sample predictions with confidence scores.

## Link (dataset):
https://www.kaggle.com/datasets/snmahsa/animal-image-dataset-cats-dogs-and-foxes/code?authuser=0

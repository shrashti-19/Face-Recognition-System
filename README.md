# Face Recognition Model

This is a machine learning project that uses a Convolutional Neural Network (CNN) to recognize faces from images using the LFW (Labeled Faces in the Wild) dataset.

## Project Overview

This project demonstrates how to build a deep learning model for **face recognition**. We used the LFW dataset to train the model, which can classify people based on their face images.

## What We've Done:
Data Preprocessing: We loaded the LFW dataset, resized the images to 64x64 pixels, and normalized them for model training.

Model Training: We built and trained a CNN model on the preprocessed data to recognize faces.

Prediction: We used the trained model to predict the identity of a person from a given image.

Evaluation: The model’s performance was evaluated based on accuracy and loss, and training/validation curves were plotted.

This project allows us to identify a person from an image of their face using deep learning techniques!
### Key Features:
- **Data Preprocessing**: Includes loading the dataset, resizing, and normalizing images.
- **Model Building**: A CNN is trained on the dataset to recognize faces.
- **Evaluation**: The model is evaluated using accuracy metrics and loss functions.
- **Prediction**: Given an image of a person’s face, the model predicts their identity.

## Dataset

The project uses the [LFW (Labeled Faces in the Wild)](http://vis-www.cs.umass.edu/lfw/) dataset. The dataset contains images of famous people and is used for benchmarking face recognition algorithms.

- **LFW Dataset Description**:
  - 13,000 labeled images of 5,749 people.
  - Images are collected from the internet and vary in lighting, pose, and occlusion.

## Installation

To run this project, follow the steps below:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Face-Recognition-Model.git
   cd Face-Recognition-Model

   
2. Install the required libraries:
   pip install -r requirements.txt
   
**Usage**
1. Data Preprocessing:
The LFW dataset is preprocessed by resizing images to 64x64 pixels and normalizing the pixel values.

2. Model Training:
The CNN model is trained using the preprocessed images, and the training and validation accuracy are plotted.

3. Face Prediction:
Once the model is trained, it can predict the identity of a person given their image.

**Evaluation**
The model is evaluated using accuracy and loss metrics, and the training and validation curves are plotted to observe the model’s performance.

**Contributing**
Feel free to fork this project, create issues, and submit pull requests. Contributions are always welcome!


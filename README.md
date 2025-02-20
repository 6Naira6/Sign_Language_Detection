# Sign Language Classification using Deep Learning

## Overview
This project implements a **Sign Language Recognition System** using a deep learning model trained on the **American Sign Language MNIST Dataset**. The goal is to classify hand gesture images into their corresponding alphabet letters.

## Dataset
We use the **American Sign Language MNIST Dataset**, which is similar to the original MNIST dataset. The dataset consists of:
- **27,455 training images**
- **7,172 testing images**
- Each image is **grayscale, 28x28 pixels**
- Labels correspond to the **English alphabet (A-Z, excluding J and Z)**

[Download the dataset from Kaggle](https://www.kaggle.com/datamunge/sign-language-mnist).

## Features
- Uses **Convolutional Neural Networks (CNNs)** for high accuracy.
- Trained on the **Sign Language MNIST dataset**.
- Supports real-time gesture classification.
- Implemented using **TensorFlow/Keras**.

## Installation
To run this project, clone the repository and install the required dependencies:

```bash
# Clone the repository
git clone https://github.com/your-username/Sign-Language-Classification.git
cd Sign-Language-Classification

# Install dependencies
pip install -r requirements.txt
```

## Usage
Run the following command to start training the model:

```bash
python train.py
```

To test the model on a sample image:

```bash
python predict.py --image path/to/image.jpg
```

## Model Architecture
The project employs a **Convolutional Neural Network (CNN)** with multiple layers including:
- **Convolutional Layers** for feature extraction
- **Max-Pooling Layers** to reduce dimensionality
- **Fully Connected Layers** for classification

## Results
The model achieves high accuracy in recognizing sign language letters, making it useful for applications in **assistive technology, human-computer interaction, and accessibility tools**.

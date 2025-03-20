# Image Captioning with ResNet and LSTM

This project demonstrates how to perform **Automatic Image Captioning** by using **ResNet** for image feature extraction and **LSTM** (Long Short-Term Memory) for sequence modeling. We leverage the **COCO 2014 dataset**, process image features, and generate descriptive captions for the given images.

## Overview

The goal of this project is to generate descriptive captions for images using a combination of deep learning models:
- **ResNet**: A convolutional neural network (CNN) used to extract features from images.
- **LSTM**: A recurrent neural network (RNN) used to generate captions based on the extracted features.
- **Word Embedding**: Words in the captions are represented using embedding layers to convert text data into numerical data that the LSTM can understand.

### 1. **COCO Dataset Download**
The **COCO 2014 dataset** is used, which contains over 80,000 images and their corresponding captions. To start, you need to download the following parts:
- **2014 Train/Val dataset**: Includes both images and annotations.
- **2014 Test dataset**: If you plan to evaluate the model on unseen data.

You can download it from the [COCO dataset website](http://cocodataset.org/#download).

### 2. **Data Preprocessing**
Data preprocessing involves preparing the image data and captions for model training:
- **Image Processing**: Images are resized and normalized using **ResNet's pre-trained weights**.
- **Text Processing**: The captions are tokenized and converted to word embeddings for feeding into the LSTM model.
- **Data Augmentation**: To enhance model generalization, we perform random image transformations (e.g., flipping, rotation).

 

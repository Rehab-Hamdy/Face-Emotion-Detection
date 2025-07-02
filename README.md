# Face-Emotion-Detection

# Real-Time Emotion Recognition System 😃🎥

A complete system that detects human emotions from facial expressions in **real time** using deep learning and machine learning techniques.

The project combines both a **Convolutional Neural Network (CNN)** and a **Support Vector Machine (SVM)** to classify emotions such as *Happy*, *Angry*, *Sad*, *Surprise*, and more based on facial features.

## 🔍 Project Overview

Two different modeling approaches were implemented and evaluated:

- 🔹 **CNN Model**: Trained on the FER2013 dataset with custom image preprocessing:
  - Noise reduction
  - CLAHE (contrast enhancement)
  - Unsharp masking (for edge clarity)

- 🔹 **SVM Model**: Trained using **HOG features** (Histogram of Oriented Gradients) to capture facial gradients and structures, producing strong results in offline classification.

For real-time emotion detection, **MTCNN** was used to detect faces from a webcam feed, and the **CNN model** was deployed to classify emotions with live confidence scores.

## 🧠 Technologies Used

- TensorFlow / Keras
- Scikit-learn
- OpenCV
- MTCNN
- Python
- HOG (Histogram of Oriented Gradients)

## 📦 Dataset

- **FER2013**: Facial Expression Recognition 2013 dataset  
  Available on [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

## 🎭 Emotion Classes

The model is trained to detect the following 7 emotions:

- 😠 **Angry**
- 🤢 **Disgust**
- 😨 **Fear**
- 😄 **Happy**
- 😢 **Sad**
- 😲 **Surprise**
- 😐 **Neutral**

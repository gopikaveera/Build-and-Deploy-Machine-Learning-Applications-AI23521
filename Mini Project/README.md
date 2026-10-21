# 🌿 GreenLeaf AI
### Plant Disease Detection Using Deep Learning

GreenLeaf AI is a deep learning application that identifies plant health and disease categories from leaf images using a custom 7-layer Convolutional Neural Network (CNN).

The application provides an interactive interface built with Streamlit, allowing users to upload leaf images or capture images using a camera for classification.

---

## 📌 Project Overview

Plant diseases can affect agricultural productivity and crop quality. Identifying visible symptoms early can help farmers and agricultural professionals investigate potential plant health problems.

GreenLeaf AI uses deep learning and computer vision to classify plant leaf images into 38 predefined plant health and disease categories.

The system uses a trained CNN model built with TensorFlow and Keras.

### 🎯 Objectives

- Develop a CNN-based plant disease classification system.
- Classify plant leaf images into predefined categories.
- Apply image preprocessing techniques before prediction.
- Display prediction labels and model confidence scores.
- Provide an interactive web interface using Streamlit.
- Explore the practical application of deep learning in agriculture.

---

## ✨ Features

- 🌱 **Deep Learning Classification:** Uses a trained 7-layer CNN.
- 🍃 **38 Output Categories:** Supports plant health and disease classification.
- 🖼️ **Image Upload:** Upload JPG, JPEG, or PNG leaf images.
- 📷 **Camera Input:** Capture leaf images for analysis.
- 📊 **Prediction Confidence:** Displays the model's output score.
- ⚡ **Interactive Interface:** Built using Streamlit.
- 🧠 **Image Preprocessing:** Resizes images to 224 × 224 pixels.
- 💻 **Local Execution:** Runs on a computer with the required dependencies installed.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | CNN model loading and inference |
| NumPy | Numerical operations |
| Pillow (PIL) | Image loading and preprocessing |
| OpenCV | Image processing |
| Streamlit | Web application interface |
| PlantVillage Dataset | Plant leaf image dataset |

---

## 🧠 Model Architecture

The project uses a custom 7-layer Convolutional Neural Network.

The architecture includes:

- Convolutional layers for extracting visual features.
- Batch normalization for stabilizing training.
- ReLU activation functions.
- Max-pooling layers for spatial downsampling.
- Dropout layers for regularization.
- Global average pooling.
- Fully connected classification layers.
- Softmax output for 38 categories.

### Model Input

- Image height: 224 pixels
- Image width: 224 pixels
- Color channels: 3 (RGB)

**Input shape:** `(224, 224, 3)`

### Model Output

The model produces 38 class scores.

**Output shape:** `(38,)` for an individual image.

The application selects the category with the highest output score.

> Note: A softmax score is not necessarily a calibrated probability of disease or a measure of real-world diagnostic reliability.

---

## 🔄 System Workflow

1. The user uploads a plant leaf image or captures one using the camera.
2. The application loads the image.
3. The image is converted to RGB format.
4. The image is resized to 224 × 224 pixels.
5. Pixel values are normalized to the range 0–1.
6. The trained CNN processes the image.
7. The model generates scores for 38 categories.
8. The category with the highest score is selected.
9. The application displays the predicted category and confidence score.

---

## 📂 Project Structure

```text
GreenLeaf-AI/
│
├── stream.py
├── Application.py
├── plant_disease_cnn7_best.keras
├── plant_disease_detection_7_layer.py
├── Plant_Disease_Detection_7_Layer-nb.ipynb
│
├── PlantVillage_DataSet/
├── PlantVillage_DataSet.zip
│
├── GifDemo.gif
├── Technical Report.docx
└── README.md

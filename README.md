# 🩺 Skin Disease Classification using MobileNetV2

This project is a deep learning-based skin disease classifier trained on the HAM10000 dataset. It uses a fine-tuned MobileNetV2 model to detect 7 types of skin diseases from images. The model is integrated into an interactive Streamlit web app for user-friendly inference.

## 🚀 Features

- Fine-tuned **MobileNetV2** on HAM10000 dataset
- Achieved **91% accuracy** on the test set
- Supports **image upload and classification** via a web interface
- Displays **confidence scores** and **disease info**
- Provides **external trusted links** (WebMD, Mayo Clinic) and **expandable sections** for user education

## 🧪 Evaluation Results

```
Test Accuracy : 91%
Test Macro F1 Score : 88%
Test Classification Report:
              precision    recall  f1-score   support

       akiec       0.98      0.86      0.91        69
         bcc       0.92      0.96      0.94        93
         bkl       0.78      0.94      0.85       228
          df       0.87      0.93      0.90        28
         mel       0.77      0.74      0.75       226
          nv       0.96      0.93      0.94      1338
        vasc       0.81      1.00      0.89        21

Overall accuracy: 0.91
```

## 🏗️ Model Training

- Base Model: `MobileNetV2`
- Fine-tuning: Last 20 layers
- Loss: Custom Focal Loss
- Optimizer: Adam with ReduceLROnPlateau
- EarlyStopping: Patience = 10
- Final Epochs: 50 (Best at Epoch 45)
- Final Training Accuracy: 95.72%
- Final Validation Accuracy: 82.13%

## 🖼️ Supported Classes

- `akiec` : Actinic Keratoses and Intraepithelial Carcinoma
- `bcc` : Basal Cell Carcinoma
- `bkl` : Benign Keratosis
- `df` : Dermatofibroma
- `mel` : Melanoma
- `nv` : Melanocytic Nevi
- `vasc` : Vascular Lesions

## 🧰 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the web app
```bash
streamlit run app.py
```

### 3. Upload an image and view predictions with details.

## 📁 Files

- `main.py` — Model training script
- `evaluate.py` — Model evaluation on test set
- `app.py` — Streamlit web interface
- `skin_disease_classifier.h5` — Trained model
- `label_encoder.pkl` — Label encoder for class names

## 📚 Dataset

[HAM10000 Dataset - Kaggle](https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000)

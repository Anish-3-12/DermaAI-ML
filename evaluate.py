import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf

# Focal loss definition (for loading the model)
def focal_loss(gamma=2., alpha=0.25):
    def focal(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        probs = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_factor = alpha * tf.pow(1. - probs, gamma)
        return focal_factor * cross_entropy
    return focal

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load metadata
df = pd.read_csv("HAM10000_metadata.csv")
DATA_DIR_1 = "HAM10000_images_part_1"
DATA_DIR_2 = "HAM10000_images_part_2"

def find_image_path(image_id):
    for folder in [DATA_DIR_1, DATA_DIR_2]:
        path = os.path.join(folder, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None

df["path"] = df["image_id"].apply(find_image_path)
df = df[df["path"].notnull()]
df["label"] = le.transform(df["dx"])

# Prepare test data
IMG_SIZE = 224
X = []
y_true = []

for _, row in df.sample(frac=0.2, random_state=42).iterrows():  # Use same val set ratio
    try:
        img = load_img(row["path"], target_size=(IMG_SIZE, IMG_SIZE))
        img_array = preprocess_input(img_to_array(img))
        X.append(img_array)
        y_true.append(row["label"])
    except:
        continue

X = np.array(X)
y_true = np.array(y_true)

# Load model with custom loss
model = load_model("skin_disease_classifier.h5", custom_objects={"focal": focal_loss()})

# Predict
y_pred_probs = model.predict(X)
y_pred = np.argmax(y_pred_probs, axis=1)

# Classification report
target_names = le.classes_
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=target_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

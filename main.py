import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, LearningRateScheduler

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

# Encode labels
le = LabelEncoder()
df["label"] = le.fit_transform(df["dx"])
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Image preprocessing
IMG_SIZE = 224
X, y = [], []
for _, row in df.iterrows():
    try:
        img = load_img(row["path"], target_size=(IMG_SIZE, IMG_SIZE))
        img_array = preprocess_input(img_to_array(img))
        X.append(img_array)
        y.append(row["label"])
    except:
        continue

X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Custom focal loss function
def focal_loss(gamma=2., alpha=0.25):
    def focal(y_true, y_pred):
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=y_pred.shape[-1])
        cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        probs = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_factor = alpha * tf.pow(1. - probs, gamma)
        return focal_factor * cross_entropy
    return focal

# Learning rate warmup
def lr_schedule(epoch, lr):
    if epoch < 5:
        return lr + 1e-4
    return lr

# Model setup
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
for layer in base_model.layers[-20:]:
    layer.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.4)(x)
output = Dense(len(np.unique(y)), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer="adam", loss=focal_loss(), metrics=["accuracy"])

# Callbacks
callbacks = [
    EarlyStopping(monitor="val_loss", patience=12, min_delta=0.001, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=1e-6, verbose=1),
    ModelCheckpoint("skin_disease_classifier.h5", save_best_only=True, monitor="val_loss", mode="min"),
    LearningRateScheduler(lr_schedule)
]

# Train
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weights,
    callbacks=callbacks
)

with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)

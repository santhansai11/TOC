"""
Train Emotion Detection CNN (MobileNetV2) on local FER dataset.

Run ONCE before starting app.py:
    python train_model.py

Dataset expected at:
    archive/train/<emotion>/  (angry, disgust, fear, happy, neutral, sad, surprise)
    archive/test/<emotion>/
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR   = os.path.join(ROOT, "archive", "train")
TEST_DIR    = os.path.join(ROOT, "archive", "test")
MODEL_PATH  = os.path.join(ROOT, "model", "emotion_model.h5")
IMG_SIZE    = 48
BATCH_SIZE  = 32
EPOCHS      = 30
CLASSES     = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
NUM_CLASSES = len(CLASSES)

os.makedirs(os.path.join(ROOT, "model"), exist_ok=True)

print(f"TensorFlow: {tf.__version__}")
print(f"Train dir : {TRAIN_DIR}")
print(f"Test dir  : {TEST_DIR}")
print(f"Model path: {MODEL_PATH}")

# Validate dataset
for split_dir in [TRAIN_DIR, TEST_DIR]:
    if not os.path.isdir(split_dir):
        print(f"❌ Missing: {split_dir}"); sys.exit(1)
    for cls in CLASSES:
        p = os.path.join(split_dir, cls)
        if not os.path.isdir(p):
            print(f"⚠  Missing class folder: {p}")

# ─── Data Generators ──────────────────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode="nearest",
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASSES,
    shuffle=True,
)

val_gen = val_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    classes=CLASSES,
    shuffle=False,
)

# Class weights to reduce neutral/majority bias
class_counts = np.bincount(train_gen.classes, minlength=NUM_CLASSES)
total = np.sum(class_counts)
class_weights = {
    i: float(total / (NUM_CLASSES * max(1, class_counts[i])))
    for i in range(NUM_CLASSES)
}
print("Class counts:", {CLASSES[i]: int(class_counts[i]) for i in range(NUM_CLASSES)})
print("Class weights:", {CLASSES[i]: round(class_weights[i], 3) for i in range(NUM_CLASSES)})

# ─── Model: MobileNetV2 Transfer Learning ────────────────────────────────────
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
)
base_model.trainable = False

inputs  = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(512, activation="relu")(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.4)(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.summary()

# ─── Phase 1: Head only ───────────────────────────────────────────────────────
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks = [
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_accuracy", verbose=1),
    EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1),
]

print("\n=== Phase 1: Training classification head ===")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights,
)

# ─── Phase 2: Fine-tune top layers ───────────────────────────────────────────
print("\n=== Phase 2: Fine-tuning top backbone layers ===")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(5e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=20,
    callbacks=callbacks,
    class_weight=class_weights,
)

print(f"\n✅ Model saved → {MODEL_PATH}")

# ─── Evaluate ─────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(val_gen)
print(f"Test Accuracy: {acc * 100:.2f}%")

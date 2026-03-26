"""
train_model.py — Train a CNN on FER2013 and export to TensorFlow.js format
===========================================================================
Usage:
    python train_model.py              # full training
    python train_model.py --smoke-test # quick architecture check only

Output:
    ../model/model.json + weight shards  (TF.js LayersModel format)
    fer2013_cnn.h5                       (Keras checkpoint)

Dataset:
    Automatically downloaded via Kaggle API:
    → msambare/fer2013  (48×48 grayscale, 7 emotion classes)

Read SETUP.md before running!
"""

import argparse
import os
import sys
import zipfile
import numpy as np

# ─── Smoke-test shortcut ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--smoke-test', action='store_true',
                    help='Just build and verify the model architecture, no training.')
args = parser.parse_args()

# ─── Imports ──────────────────────────────────────────────────────────────────
print("[1/7] Importing libraries...")
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflowjs as tfjs
from PIL import Image

print(f"     TensorFlow {tf.__version__}")
print(f"     GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# ─── Constants ────────────────────────────────────────────────────────────────
IMG_SIZE    = 48
NUM_CLASSES = 7
BATCH_SIZE  = 64
EPOCHS      = 50
EMOTIONS    = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# Folder emitted by Kaggle ZIP: images/train/<emotion>/ and images/test/<emotion>/
DATASET_DIR  = os.path.join(os.path.dirname(__file__), 'fer2013_data')
TRAIN_DIR    = os.path.join(DATASET_DIR, 'train')
TEST_DIR     = os.path.join(DATASET_DIR, 'test')
MODEL_OUT_H5 = os.path.join(os.path.dirname(__file__), 'fer2013_cnn.h5')
MODEL_OUT_JS = os.path.join(os.path.dirname(__file__), '..', 'model')


# ─── Step 1: Download dataset from Kaggle ─────────────────────────────────────
def download_dataset():
    if os.path.isdir(TRAIN_DIR):
        print("[2/7] Dataset already present, skipping download.")
        return

    print("[2/7] Downloading FER2013 from Kaggle...")
    try:
        import kaggle  # requires ~/.kaggle/kaggle.json
    except ImportError:
        print("ERROR: 'kaggle' package not installed. Run: pip install -r requirements.txt")
        sys.exit(1)

    os.makedirs(DATASET_DIR, exist_ok=True)
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        'msambare/fer2013',
        path=DATASET_DIR,
        unzip=True,
        quiet=False
    )

    # The kaggle dataset 'msambare/fer2013' extracts directly as train/ and test/
    # Verify structure
    if not os.path.isdir(TRAIN_DIR):
        print(f"ERROR: Expected folder not found: {TRAIN_DIR}")
        print("       Check that the dataset extracted correctly.")
        sys.exit(1)

    print(f"     Dataset ready at: {DATASET_DIR}")


# ─── Step 2: Build the CNN ────────────────────────────────────────────────────
def build_model():
    """
    Lightweight but effective CNN for 48×48 grayscale emotion classification.
    Architecture: 4 Conv blocks → GlobalAvgPool → Dense → Softmax
    ~1.2M parameters — small enough for TF.js browser loading.
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='input')

    # Block 1
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.35)(x)

    # Block 4
    x = layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.4)(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name='output')(x)

    model = keras.Model(inputs, outputs, name='FER2013_CNN')
    return model


# ─── Step 3: Data generators ──────────────────────────────────────────────────
def make_generators():
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    train_aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    val_aug = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_aug.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    val_gen = val_aug.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen


# ─── Step 4: Train ────────────────────────────────────────────────────────────
def train(model, train_gen, val_gen):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_OUT_H5, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    print(f"\n[4/7] Training for up to {EPOCHS} epochs (early stopping enabled)...")
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    return history


# ─── Step 5: Evaluate ─────────────────────────────────────────────────────────
def evaluate(model, val_gen):
    print("\n[5/7] Evaluating on test set...")
    loss, acc = model.evaluate(val_gen, verbose=0)
    print(f"     Test Loss:     {loss:.4f}")
    print(f"     Test Accuracy: {acc * 100:.2f}%")

    # Per-class breakdown
    val_gen.reset()
    y_pred = np.argmax(model.predict(val_gen, verbose=0), axis=1)
    y_true = val_gen.classes

    print("\n     Per-class accuracy:")
    for i, emotion in enumerate(EMOTIONS):
        mask = (y_true == i)
        if mask.sum() > 0:
            cls_acc = (y_pred[mask] == i).sum() / mask.sum()
            bar = '█' * int(cls_acc * 20)
            print(f"       {emotion:<12} {bar:<20} {cls_acc*100:5.1f}%")

    return acc


# ─── Step 6: Export to TF.js ──────────────────────────────────────────────────
def export_tfjs(model):
    print(f"\n[6/7] Exporting to TF.js format → {MODEL_OUT_JS}")
    os.makedirs(MODEL_OUT_JS, exist_ok=True)
    tfjs.converters.save_keras_model(model, MODEL_OUT_JS)
    print("     Export complete!")
    print(f"     Files written:")
    for f in os.listdir(MODEL_OUT_JS):
        size = os.path.getsize(os.path.join(MODEL_OUT_JS, f)) / 1024
        print(f"       {f}  ({size:.1f} KB)")


# ─── Step 7: Summary ──────────────────────────────────────────────────────────
def print_summary(acc):
    print("\n" + "═" * 60)
    print("  ✅  TRAINING COMPLETE")
    print("═" * 60)
    print(f"  Test Accuracy : {acc * 100:.2f}%")
    print(f"  Keras model   : {MODEL_OUT_H5}")
    print(f"  TF.js model   : {os.path.abspath(MODEL_OUT_JS)}/")
    print()
    print("  Next steps:")
    print("  1. Open index.html in a browser (via Live Server)")
    print("  2. The app will auto-detect model/model.json")
    print("  3. Look for '🟢 Custom ML Model Active' badge")
    print("═" * 60)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    if args.smoke_test:
        print("\n[SMOKE TEST] Building model and running a single forward pass...")
        model = build_model()
        model.summary()
        dummy = np.random.rand(1, IMG_SIZE, IMG_SIZE, 1).astype('float32')
        out = model(dummy)
        assert out.shape == (1, NUM_CLASSES), f"Bad output shape: {out.shape}"
        probs = out.numpy()[0]
        assert abs(probs.sum() - 1.0) < 1e-5, "Probabilities don't sum to 1"
        print("\n✅  Smoke test passed — model builds and forward-pass works")
        print(f"    Output shape : {out.shape}")
        print(f"    Probabilities: {[f'{p:.3f}' for p in probs]}")
        sys.exit(0)

    download_dataset()

    print("\n[3/7] Building model...")
    model = build_model()
    model.summary()

    train_gen, val_gen = make_generators()
    print(f"     Train samples : {train_gen.samples}")
    print(f"     Val samples   : {val_gen.samples}")
    print(f"     Class mapping : {train_gen.class_indices}")

    history = train(model, train_gen, val_gen)
    acc = evaluate(model, val_gen)
    export_tfjs(model)
    print_summary(acc)


if __name__ == '__main__':
    main()

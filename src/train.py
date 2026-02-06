import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# -------------------------
# CONFIG
# -------------------------
IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 15

train_dir = "../dataset/train"
val_dir = "../dataset/validation"

# -------------------------
# DATA LOADERS
# -------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode="grayscale",
    class_mode="categorical",
    batch_size=BATCH_SIZE
)

NUM_CLASSES = train_data.num_classes
print("Classes:", train_data.class_indices)

# -------------------------
# MODEL DEFINITION
# -------------------------

# 1️⃣ Proper Keras input (grayscale)
inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 1))

# 2️⃣ Convert grayscale → 3-channel
x = Conv2D(3, (1, 1), padding="same", name="gray_to_rgb")(inputs)

# 3️⃣ MobileNetV2 backbone
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False  # Transfer learning (Phase 1 safe)

x = base_model(x)

# 4️⃣ Classification head
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

# -------------------------
# COMPILE
# -------------------------
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# TRAIN
# -------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# -------------------------
# SAVE
# -------------------------
model.save("../models/opti_fab_model.h5")
print("✅ Model saved successfully")

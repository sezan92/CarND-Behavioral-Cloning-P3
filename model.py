#!/usr/bin/env python
# coding: utf-8
import cv2
from keras.models import Sequential
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import rmsprop
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import set_random_seed

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# dynamically grow the memory used on the GPU
config.log_device_placement = True
# to log device placement (on which device the operation ran)
sess = tf.Session(config=config)
set_session(sess)

SEED = 1
np.random.seed(SEED)
set_random_seed(SEED)

DATA_FOLDER = "../data/Simulator/"

normal = os.path.join(DATA_FOLDER, "CenterLaneDriving")
recovery = os.path.join(DATA_FOLDER, "RecoveryLap")
recovery2 = os.path.join(DATA_FOLDER, "RecoveryLap2")
difficult = os.path.join(DATA_FOLDER, "Difficult")
difficult2 = os.path.join(DATA_FOLDER, "Difficult2")

names = [
    "Center Image",
    "Left Image",
    "Right Image",
    "Steering",
    "Throttle",
    "Brake",
    "Speed",
]

normal_driving_log = pd.read_csv(os.path.join(normal, "driving_log.csv"), names=names)
normal_driving_log.head()
normal_driving_log["Steering"].hist(bins=10)
recovery_driving_log = pd.read_csv(
    os.path.join(recovery, "driving_log.csv"), names=names
)
recovery_driving_log2 = pd.read_csv(
    os.path.join(recovery2, "driving_log.csv"), names=names
)
recovery_driving_log.head()
recovery_driving_log[recovery_driving_log["Steering"] != 0.0]["Steering"].hist()
recovery_driving_log2[recovery_driving_log2["Steering"] != 0.0]["Steering"].hist()
recovery_driving_log = recovery_driving_log[recovery_driving_log["Steering"] != 0.0]
recovery_driving_log2 = recovery_driving_log2[recovery_driving_log2["Steering"] != 0.0]
difficult_driving_log = pd.read_csv(
    os.path.join(difficult, "driving_log.csv"), names=names
)
difficult_driving_log["Steering"].hist()
difficult_driving_log2 = pd.read_csv(
    os.path.join(difficult2, "driving_log.csv"), names=names
)
difficult_driving_log2["Steering"].hist()
full_driving_log = pd.concat(
    [
        normal_driving_log,
        recovery_driving_log,
        recovery_driving_log2,
        difficult_driving_log,
        difficult_driving_log2,
    ]
)
full_difficult_driving_log = pd.concat([difficult_driving_log, difficult_driving_log2])
data = full_difficult_driving_log[["Center Image", "Steering"]]
data["Steering"].hist()
data = data[data["Steering"] != 0.0]
train, test = train_test_split(data)
train["Steering"].hist()
test["Steering"].hist()


def read_images(batch):
    images = [None] * len(batch)
    for i, image_name in enumerate(batch):
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images[i] = image
    images = np.array(images)
    return images


def loader(data, batch_size):
    i = 0
    while True:
        batch = data.iloc[i : i + batch_size]
        batch_image_names = batch["Center Image"].values.tolist()
        batch_images = read_images(batch_image_names)
        steering = batch["Steering"].values.tolist()
        i = i + batch_size
        if i >= len(data):
            i = 0
        yield batch_images, steering


model = Sequential()
model.add(layers.Lambda(lambda x: x / 255.0, input_shape=(160, 320, 3)))
model.add(layers.Cropping2D(cropping=((50, 20), (0, 0))))
model.add(
    layers.Conv2D(
        64, (5, 5), strides=(1, 1), activation="relu", kernel_regularizer=l2(0.01)
    )
)
model.add(layers.Dropout(0.5))
model.add(layers.MaxPooling2D(pool_size=2, strides=2))
model.add(layers.Conv2D(128, (5, 5), activation="relu", kernel_regularizer=l2(0.01)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (5, 5), activation="relu", kernel_regularizer=l2(0.01)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(units=250, activation="relu", kernel_regularizer=l2(0.01)))
model.add(layers.Dense(units=120, activation="relu", kernel_regularizer=l2(0.01)))
model.add(layers.Dense(units=84, activation="relu", kernel_regularizer=l2(0.01)))
model.add(layers.Dense(1))
model.compile(optimizer=rmsprop(lr=0.0001, rho=0.9), loss="mse")
model.summary()

train_loader = loader(train, 32)
test_loader = loader(test, 32)

history = model.fit_generator(
    train_loader,
    steps_per_epoch=len(train) // 32,
    validation_data=test_loader,
    validation_steps=87,
    epochs=50,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=5),
        ModelCheckpoint(
            "difficult_driving_model.h5",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
    ],
)

plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.legend()
plt.show()

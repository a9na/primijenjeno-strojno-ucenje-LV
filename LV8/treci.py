from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import Rescaling

train_ds = image_dataset_from_directory(
    directory='/home/profesor/Downloads/dataset/Train',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(48, 48))

test_ds = image_dataset_from_directory(
    directory='/home/profesor/Downloads/dataset/TestSorted',
    labels='inferred',
    label_mode='categorical',
    batch_size=32,
    image_size=(48, 48))

model = keras.Sequential(
    [
        keras.Input(shape=(48,48,3)),
        layers.experimental.preprocessing.Rescaling(scale=1./255),
        layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
        layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(rate=0.2),
        layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        layers.Conv2D(32, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(rate=0.2),
        layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
        layers.Conv2D(128, kernel_size=(3,3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Dropout(rate=0.2),
        layers.Flatten(),
        layers.Dense(43, activation='softmax')
    ]
)


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_ds, epochs=5, batch_size=32, validation_data=test_ds)

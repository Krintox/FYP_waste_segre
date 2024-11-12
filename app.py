import os
import math
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
import keras.utils as ku
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set directories
dir_example = "E:/Bunny/Final_Year_Project/Envier-portal/waste_segre/Data"
train_dir = "E:/Bunny/Final_Year_Project/Envier-portal/waste_segre/Data/Train"
test_dir = "E:/Bunny/Final_Year_Project/Envier-portal/waste_segre/Data/Test"
dir_with_examples = 'E:/Bunny/Final_Year_Project/Envier-portal/waste_segre/visualize'

# List class names
classes = os.listdir(dir_example)
print(classes)

train_classes = os.listdir(train_dir)
print(train_classes)

# Visualize some example images
# Visualize some example images
files_per_row = 6
files_in_dir = os.listdir(dir_with_examples)
number_of_cols = files_per_row
number_of_rows = math.ceil(len(files_in_dir) / files_per_row)

fig, axs = plt.subplots(number_of_rows, number_of_cols)
fig.set_size_inches(20, 15, forward=True)

# Ensure axs is always 2D
if number_of_rows == 1:
    axs = np.expand_dims(axs, axis=0)  # Add a new dimension if there's only 1 row
if number_of_cols == 1:
    axs = np.expand_dims(axs, axis=1)  # Add a new dimension if there's only 1 column

for i in range(0, len(files_in_dir)):
    file_name = files_in_dir[i]
    image = Image.open(f'{dir_with_examples}/{file_name}')
    row = i // files_per_row
    col = i % files_per_row
    axs[row, col].imshow(image)
    axs[row, col].axis('off')

plt.show()


# Setup ImageDataGenerators
train_generator = ImageDataGenerator(rescale=1/255)
train_generator = train_generator.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='sparse'
)

labels = train_generator.class_indices
print(labels)
labels = dict((v, k) for k, v in labels.items())
print(labels)

# Test Data Generator
test_generator = ImageDataGenerator(rescale=1/255)
test_generator = test_generator.flow_from_directory(
    test_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='sparse'
)

test_labels = test_generator.class_indices
print(test_labels)
test_labels = dict((v, k) for k, v in test_labels.items())
print(test_labels)

# Create Model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(300, 300, 3), activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(6, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(train_generator, epochs=10, steps_per_epoch=2184 // 32)

# Image Prediction
test_img = 'Trash-Classifier-in-Python-using-Tensorflow/Data/Test/paper/paper522.jpg'
img = ku.load_img(test_img, target_size=(300, 300))
img = ku.img_to_array(img, dtype=np.uint8)
img = np.array(img) / 255.0
prediction = model.predict(img[np.newaxis, ...])

print("Probability:", np.max(prediction[0], axis=-1))
predicted_class = labels[np.argmax(prediction[0], axis=-1)]
print("Classified:", predicted_class)

plt.axis('off')
plt.imshow(img.squeeze())
plt.title("Loaded Image")

# Save Model
model.save('modelnew.h5')

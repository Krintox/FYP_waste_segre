# Importing TensorFlow
import os
import matplotlib.pyplot as plt
from PIL import Image
import math
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout, SpatialDropout2D
from tensorflow.keras.losses import sparse_categorical_crossentropy, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Using CPU
device = "/device:CPU:0"

# Data Importing
dir_example = "E:/Bunny/Final_Year_Project/Envier-portal/waste_segre/Data"
classes = os.listdir(dir_example)
print(classes)  # Output the classes

dir_example = "E:/Bunny/Final_Year_Project/Envier-portal/waste_segre/Data/Train"
train_classes = os.listdir(dir_example)
print(train_classes)  # Output the training classes

# Data Visualization
dir_with_examples = 'E:/Bunny/Final_Year_Project/Envier-portal/waste_segre/visualize'
files_per_row = 6
files_in_dir = os.listdir(dir_with_examples)
number_of_cols = files_per_row
number_of_rows = int(len(files_in_dir) / number_of_cols)

# Generate the subplots
fig, axs = plt.subplots(number_of_rows, number_of_cols)
fig.set_size_inches(20, 15, forward=True)

# Map each file to subplot
try:
    for i in range(0, len(files_in_dir)):
        file_name = files_in_dir[i]
        image = Image.open(f'{dir_with_examples}/{file_name}')
        row = math.floor(i / files_per_row)
        col = i % files_per_row
        axs[row, col].imshow(image)
        axs[row, col].axis('off')
except Exception as e:
    print(f"Error displaying images: {e}")

# Show the plot
plt.show()

# Preparing Data
train = 'E:/Bunny/Final_Year_Project/Envier-portal/waste_segre/Data/Train'
test = 'E:/Bunny/Final_Year_Project/Envier-portal/waste_segre/Data/Test'
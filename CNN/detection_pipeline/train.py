import os
import tensorflow as tf
from data_processing import load_image

# Limit the GPU growth
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'{len(gpus)} GPUs found')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

## Load the training data
DIR_PATH = os.getcwd()
IMG_PATH = os.path.join(DIR_PATH, 'data_aug\\images')
images = tf.data.Dataset.list_files(os.path.join(IMG_PATH,'*.jpg'),shuffle=False)
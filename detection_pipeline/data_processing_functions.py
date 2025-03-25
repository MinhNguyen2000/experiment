''' This script includes the function to load images from the file path and split the data into training, validation, and testing sets. '''

import os
import tensorflow as tf 
import json         # For parsing the json label files
import numpy as np
import random
from matplotlib import pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'{len(gpus)} GPUs found')

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

DIR_PATH = os.getcwd()
IMG_RAW_PATH = os.path.join(DIR_PATH, 'data_raw\\images')

def load_image(x):
    '''Function to load image from file path. This function reads the byte data from the file path and decodes it into an image tensor'''
    byte_img = tf.io.read_file(x)
    img = tf.io.decode_jpeg(byte_img)
    return img

def count_images(directories):
    """
    Function to count the number of images in a list of directories
    """
    return [len(tf.io.gfile.glob(os.path.join(directory, '*.jpg'))) for directory in directories]

def move_images(source_dir, target_dir, num_images):
    """
    Function to move images from one directory to another
    """
    images = tf.io.gfile.glob(os.path.join(source_dir, '*.jpg'))        # List of image paths in the source directory
    random.shuffle(images)
    for image_path in images[:num_images]:
        filename = os.path.basename(image_path)                         # Get the filename    
        target_path = os.path.join(target_dir, filename)                # Create the target path
        tf.io.gfile.copy(image_path, target_path, overwrite=True)       # Copy the image to the target path
        tf.io.gfile.remove(image_path)                                  # Remove the image from the source directory

def data_split(train_dir, val_dir, test_dir, desired_train=0.8, desired_val=0.2, seed=42):
    """
    Function to split the data (both images and labels) into training, validation, and possibly testing sets
    """
    # Calculate the proportion of images for training, validation, and testing
    total_proportion = desired_train + desired_val
    if total_proportion > 1:
        raise ValueError('The sum of desired_train and desired_val must not exceed 1')
    else:
        desired_test = 1 - total_proportion

    train_img_dir = os.path.join(train_dir, 'images')
    # Ensure the validation image directory exists
    val_img_dir = os.path.join(val_dir, 'images')
    os.makedirs(val_img_dir, exist_ok=True)
    # Ensure the test image directory exists (if provided)
    test_img_dir = os.path.join(test_dir, 'images')
    os.makedirs(test_img_dir, exist_ok=True)

    # Find the current number of images in train, val, and test
    train_count, val_count, test_count = count_images([train_img_dir,val_img_dir,test_img_dir])
    total_images = train_count + val_count + test_count
    print(total_images)

    # Find the desired number of images in train, val, and test
    desired_train_count = int(total_images * desired_train)
    desired_val_count = int(total_images * desired_val) if desired_test else total_images - desired_train_count
    desired_test_count = total_images - desired_train_count - desired_val_count if desired_test else 0
    
    # Moving images between validation and test directories, then recount the images
    if val_count > desired_val_count:
        move_images(val_img_dir, train_img_dir, val_count - desired_val_count)
    elif val_count < desired_val_count:
        move_images(train_img_dir, val_img_dir, desired_val_count - val_count)

    train_count, val_count, test_count = count_images([train_img_dir,val_img_dir,test_img_dir])

    # Moving the images between the training and test directories
    if test_count > desired_test_count:
        move_images(test_img_dir, train_img_dir, test_count - desired_test_count)
    elif test_count < desired_test_count:
        move_images(train_img_dir, test_img_dir, desired_test_count - test_count)

    # Find the current number of images in train, val, and test after the shift
    train_count, val_count, test_count = count_images([train_img_dir,val_img_dir,test_img_dir])

    for folder in [train_dir,test_dir,val_dir]:
        for file in os.listdir(os.path.join(folder,'images')):
            filename = file.split('.')[0]+'.json'                               # JSON file of the label
            existing_filepath = os.path.join(train_dir,'labels',filename)

            if os.path.exists(existing_filepath):
                new_filepath = os.path.join(folder,'labels',filename)
                os.replace(existing_filepath, new_filepath)

    return train_count, val_count, test_count


# ## Testing the data loading function

# # Load Images into the TF Data pipeline
# images = tf.data.Dataset.list_files(os.path.join(IMG_RAW_PATH,'*.jpg'),shuffle=False)

# # Map the function onto the dataset
# images = images.map(load_image)

# # Look at one image sample and its shape
# print('Image pixel values \n',images.as_numpy_iterator().next())
# print('\nImage shape ', images.as_numpy_iterator().next().shape)

# image_generator = images.batch(4).as_numpy_iterator()

# plot_images = image_generator.next()    # Get the next batch of images
# fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
# for idx, image in enumerate(plot_images):
#     ax[idx%2, idx//2].imshow(image)
#     ax[idx%2, idx//2].axis('off')

# plt.show()


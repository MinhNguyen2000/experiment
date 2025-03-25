from data_processing_functions import load_image, data_split
import os


## Split the data into training, validation, and testing sets in the data_sorted folder
DIR_PATH = os.getcwd()
DATA_SORTED_PATH = os.path.join(DIR_PATH, 'data_sorted')
train_dir = os.path.join(DATA_SORTED_PATH,'train')
val_dir = os.path.join(DATA_SORTED_PATH,'val')
test_dir = os.path.join(DATA_SORTED_PATH,'test')

data_split(train_dir, val_dir, test_dir, 
           desired_train=0.7, 
           desired_val=0.15)
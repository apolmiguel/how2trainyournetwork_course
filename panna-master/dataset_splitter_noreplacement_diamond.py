import numpy as np 
import os
import shutil
import sys

## NOTE: Run in root ## 

# User-called variables
split_boundaries = [0,100, 500, 1000, 5000] # dataset partitioning 
val_split = 0.2 # 20% of the train set will be used for val

# Directory and file variables
omegaset_dir = '/leonardo/pub/userexternal/atan0000/per_francesca/datasets/diamond5000/'
outputmain_dir = './datasets/'
dataset_tag = 'diamond'

# Create output subdirectories, with train and val subsubdirectories
for split in split_boundaries[1:]:
    output_dir = outputmain_dir + dataset_tag + '_' + str(split) + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir + 'train/'):
        os.makedirs(output_dir + 'train/')
    if not os.path.exists(output_dir + 'val/'):
        os.makedirs(output_dir + 'val/')

# Create a list of all the files in omegaset_dir
omegalist = os.listdir(omegaset_dir)
omegalist_picked = omegalist.copy() # List copy where selections will be popped.

train_list = []
val_list = []

# For each split, shuffle the fileorder and split it into train and val sets
for i in np.arange(0, len(split_boundaries)-1):
    # Obtain the current splice to be distributed
    print('Now creating the dataset for size ' + str(split_boundaries[i+1]) + '...')
    print('Taking a splice of length ' + str(split_boundaries[i+1] - split_boundaries[i]) + '...')
    start_ind = split_boundaries[i]
    end_ind = split_boundaries[i+1]
    full_pile = fileorder[start_ind:end_ind]
    # Randomize the order of the files, and split them to train and val sets
    print('\nRandomizing the order of the files...')
    np.random.shuffle(full_pile)
    split_index = int(len(full_pile) * val_split)
    train_pile = full_pile[split_index:]
    val_pile = full_pile[:split_index]
    # Add the new piles into the existing lists, to make sure larger sets contain smaller ones
    train_list.extend(train_pile)
    val_list.extend(val_pile)
    print('\nCurrent train_list length: ' + str(len(train_list)))
    print('Current val_list length: ' + str(len(val_list))+'\n\n')
    # select the files from omegalist_picked based on the current lists
    for file in train_list:
        shutil.copy(omegaset_dir + file, outputmain_dir + dataset_tag + '_' + str(split_boundaries[i+1]) + '/train/')
    for file in val_list:
        shutil.copy(omegaset_dir + file, outputmain_dir + dataset_tag + '_' + str(split_boundaries[i+1]) + '/val/')
print('Files have been distributed to the train and val sets.')



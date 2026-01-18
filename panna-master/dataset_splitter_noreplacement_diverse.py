import numpy as np 
import os
import shutil

## NOTE: Run in root ## 

# User-called variables
split_boundaries = [100, 500, 1000, 5000] # dataset partitioning 
split_labels = ['100', '500', '1k', '5k']
val_split = 0.2 # 20% of the dataset will be used for val

# Directory and file variables
omegaset_dir = '/leonardo_scratch/large/userexternal/atan0000/Carbon_full/Tr5k_n/'
diverse_superset_dir = '/leonardo_scratch/large/userexternal/atan0000/Carbon_full/'
outputmain_dir = './datasets/'
dataset_tag = 'diverse'

# Create output subdirectories, with train and val subdirectories
for split in split_boundaries:
    output_dir = outputmain_dir + dataset_tag + '_' + str(split) + '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir + 'train/'):
        os.makedirs(output_dir + 'train/')
    if not os.path.exists(output_dir + 'val/'):
        os.makedirs(output_dir + 'val/')

# Create lists and counter
omegalist = os.listdir(omegaset_dir)
train_list = []; val_list = []
dset_diff = [100,400,500,4000] # Difference between dataset sizes 
ndset_counter = 0
for i,diff in enumerate(dset_diff):
    ndset_counter += diff
    print('Now creating the dataset for size ' + str(ndset_counter) + '...')
    subdir = diverse_superset_dir + 'Tr'+split_labels[i]+'_n/'
    subdir_files = os.listdir(subdir)
    subdir_files = np.setdiff1d(subdir_files, train_list + val_list) # Remove already selected files previously
    subdir_shuffled = np.random.permutation(subdir_files)

    # Split to train and val, then copy to the output directories 
    split_index = int(len(subdir_shuffled) * val_split)
    train_pile = subdir_shuffled[split_index:]
    val_pile = subdir_shuffled[:split_index]
    train_list.extend(train_pile)
    val_list.extend(val_pile)
    print('Current train_list length: ' + str(len(train_list)))
    print('Current val_list length: ' + str(len(val_list))+'\n\n')
    for file in train_list:
        shutil.copy(subdir + file, outputmain_dir + dataset_tag + '_' + str(ndset_counter) + '/train/')
    for file in val_list:
        shutil.copy(subdir + file, outputmain_dir + dataset_tag + '_' + str(ndset_counter) + '/val/')

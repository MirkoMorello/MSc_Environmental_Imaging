from tqdm import tqdm
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

npy_folder = './numpys_augmented_no_rotations'
reduced_size = (128, 128)
datapath_augmented = 'augmented_data_no_rotations'
if os.path.isdir(npy_folder):
    print('Numpy folder already exists')
else:
    os.mkdir(npy_folder)
    print('Numpy folder created')

photos, labels = list(), list()
labels_list = os.listdir(datapath_augmented)
for folder_label in labels_list:
    path_folder = os.path.join(datapath_augmented, folder_label)
    print(f"processing {path_folder}")
    
    output = folder_label
    file_list = os.listdir(path_folder)
    for file in tqdm(file_list):
        photo = load_img(os.path.join(path_folder, file), target_size=(128, 128))
        photo = img_to_array(photo)

        photos.append(photo)
        labels.append(output)

photos = np.asarray(photos)
labels = np.asarray(labels)
print(photos.shape, labels.shape)

np.save(os.path.join(npy_folder, 'photos.npy'), photos)
np.save(os.path.join(npy_folder, 'labels.npy'), labels)
print('Saved')
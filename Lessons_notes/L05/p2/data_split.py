import os
import shutil
import random
from tqdm import tqdm

# Define the paths
data_dir = '/home/omirako/Documents/Magistrale/Environment/datasets/augmented_classification_images'
train_dir = 'train_data'
test_dir = 'test_data'
val_dir = 'val_data'

# Create train, test, and validation directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

class_dirs = os.listdir(data_dir)
for class_dir in tqdm(class_dirs):
    class_path = os.path.join(data_dir, class_dir)
    if os.path.isdir(class_path):
        # Create subdirectories in train, test, and validation directories
        train_class_dir = os.path.join(train_dir, class_dir)
        os.makedirs(train_class_dir, exist_ok=True)
        
        test_class_dir = os.path.join(test_dir, class_dir)
        os.makedirs(test_class_dir, exist_ok=True)
        
        val_class_dir = os.path.join(val_dir, class_dir)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Get list of image filenames in class directory
        images = os.listdir(class_path)
        random.shuffle(images)
        
        # Split images into train, test, and validation sets
        num_images = len(images)
        num_train = int(0.7 * num_images)  # 70% for train
        num_test = int(0.2 * num_images)   # 20% for test
        num_val = num_images - num_train - num_test  # Remaining for validation
        
        train_images = images[:num_train]
        test_images = images[num_train:num_train+num_test]
        val_images = images[num_train+num_test:]
        
        # Move images to their respective directories
        for image in train_images:
            src = os.path.join(class_path, image)
            dst = os.path.join(train_class_dir, image)
            shutil.copy(src, dst)
        
        for image in test_images:
            src = os.path.join(class_path, image)
            dst = os.path.join(test_class_dir, image)
            shutil.copy(src, dst)
            
        for image in val_images:
            src = os.path.join(class_path, image)
            dst = os.path.join(val_class_dir, image)
            shutil.copy(src, dst)

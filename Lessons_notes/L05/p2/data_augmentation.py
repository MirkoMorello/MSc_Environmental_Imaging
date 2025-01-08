import tensorflow as tf
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm

datapath = '/home/omirako/Documents/Magistrale/Environment/datasets/classification_images'

def rotate_image(image, angle):
    angle_rad = tf.constant(angle * (3.14159 / 180.0), dtype=tf.float32)
    rotated_image = tf.image.rot90(image, k=tf.cast(angle_rad / (3.14159 / 2), tf.int32))
    
    return rotated_image

# data augmentation
# dirs = os.listdir(datapath)
rotations = 20
output_dir = '/home/omirako/Documents/Magistrale/Environment/datasets/augmented_classification_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for dir in os.listdir(datapath):
    print(f"Augmenting {dir}")
    files = os.listdir(os.path.join(datapath, dir))

    out_sub_dir = os.path.join(output_dir, dir)
    if not os.path.exists(os.path.join(output_dir, dir)):
        os.makedirs(os.path.join(output_dir, dir))

    for file in tqdm(files):
        img = load_img(os.path.join(datapath, dir, file))
        img.save(os.path.join(out_sub_dir, file))
        data = img_to_array(img)

        # horizontal flip
        flipped = tf.image.flip_left_right(data)
        flipped = tf.image.convert_image_dtype(flipped, tf.uint8)
        img = Image.fromarray(flipped.numpy())
        img.save(os.path.join(out_sub_dir, 'flipped_lr_' + file))

        # vertical flip
        flipped = tf.image.flip_up_down(data)
        flipped = tf.image.convert_image_dtype(flipped, tf.uint8)
        img = Image.fromarray(flipped.numpy())
        img.save(os.path.join(out_sub_dir, 'flipped_ud_' + file))

        # 10 rotations of 36 degrees each
        for i in range(1, rotations):
            rotated = rotate_image(data, i*36)
            rotated = tf.image.convert_image_dtype(rotated, tf.uint8)
            img = Image.fromarray(rotated.numpy())
            img.save(os.path.join(out_sub_dir, f'rotated_{i*(360/rotations)}_' + file))

        # zoom crop
        cropped = tf.image.random_crop(data, size=[128, 128, 3])
        cropped = tf.image.convert_image_dtype(cropped, tf.uint8)
        img = Image.fromarray(cropped.numpy())
        img.save(os.path.join(out_sub_dir, 'cropped_' + file))
    
        






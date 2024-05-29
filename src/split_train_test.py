import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Original data directory
data_dir = 'data/zalo/new_zalo_yolo'
fake_dir = os.path.join(data_dir, 'fake')
real_dir = os.path.join(data_dir, 'real')

# New directories for split data
output_root = 'data/new_split'
train_dir = os.path.join(output_root, 'face_train')
test_dir = os.path.join(output_root, 'face_test')
valid_dir = os.path.join(output_root, 'face_valid')

# Create new directories
for split_dir in [train_dir, test_dir, valid_dir]:
    os.makedirs(os.path.join(split_dir, 'real'), exist_ok=True)
    os.makedirs(os.path.join(split_dir, 'fake'), exist_ok=True)

# Helper function to collect all image paths from a directory
def collect_image_paths(dir_path):
    image_paths = []
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(('.jpg', '.png')):
                image_paths.append(os.path.join(root, file))
    return image_paths

# Helper function to split and copy data by image paths
def split_and_copy(image_paths, label, train_ratio=0.7, valid_ratio=0.2):
    random.shuffle(image_paths)
    train_size = int(len(image_paths) * train_ratio)
    valid_size = int(len(image_paths) * valid_ratio)
    train_data = image_paths[:train_size]
    valid_data = image_paths[train_size:train_size + valid_size]
    test_data = image_paths[train_size + valid_size:]

    # Copy to train directory
    for img_path in train_data:
        dst_dir = os.path.join(train_dir, label, os.path.relpath(os.path.dirname(img_path), start=os.path.join(data_dir, label)))
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(img_path, dst_dir)
    
    # Copy to validation directory
    for img_path in valid_data:
        dst_dir = os.path.join(valid_dir, label, os.path.relpath(os.path.dirname(img_path), start=os.path.join(data_dir, label)))
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(img_path, dst_dir)
    
    # Copy to test directory
    for img_path in test_data:
        dst_dir = os.path.join(test_dir, label, os.path.relpath(os.path.dirname(img_path), start=os.path.join(data_dir, label)))
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(img_path, dst_dir)

# Collect all image paths for fake and real videos
fake_images = collect_image_paths(fake_dir)
real_images = collect_image_paths(real_dir)

# Split and copy fake images
split_and_copy(fake_images, 'fake')

# Split and copy real images
split_and_copy(real_images, 'real')

print("Data split and copy complete.")

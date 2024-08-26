import os
import shutil
import random

data_dir = 'clothing-dataset-small-master'
train_dir = os.path.join(data_dir, 'train')
val_dir = os.path.join(data_dir, 'validation')
test_dir = os.path.join(data_dir, 'test')
class_names = ['pants', 't-shirt', 'skirt', 'dress', 'shorts', 'shoes', 'hat', 'longsleeve', 'outwear', 'shirt']

train_prop = 0.8
val_prop = 0.1
test_prop = 0.1

for new_dir in [train_dir, val_dir, test_dir]:
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

for name in class_names:
    director = os.path.join(train_dir, name)
    namef = os.listdir(director)
    random.shuffle(namef)
    num_test = len(namef) - int(len(namef) * train_prop)- int(len(namef) * val_prop)
    for i, file_name in enumerate(namef):
        src_path = os.path.join(director, file_name)
        if i < int(len(namef) * train_prop):
            dst_dir = os.path.join(train_dir, name)
        elif i < int(len(namef) * train_prop) + int(len(namef) * val_prop):
            dst_dir = os.path.join(val_dir, name)
        else:
            dst_dir = os.path.join(test_dir, name)
        dst_path = os.path.join(dst_dir, file_name)
        if not os.path.exists(dst_path) or not os.path.samefile(src_path, dst_path):
            shutil.copy(src_path, dst_path)

import numpy as np
import os
import csv
from PIL import Image

# 加载 .npz 文件
npz_file = np.load('dataset_npz/chestmnist_64.npz')

# 获取文件中的所有数组名称
data_arrays = npz_file.files  # ['train_images', 'train_labels', 'val_images', 'val_labels', 'test_images', 'test_labels']

# 创建存放图像和标签的主文件夹
output_dir = 'dataset_picture/chestmnist_64'
picture_dir = os.path.join(output_dir, 'picture_64')
split_dir = os.path.join(output_dir, 'split_64')
os.makedirs(picture_dir, exist_ok=True)
os.makedirs(split_dir, exist_ok=True)

# 创建三个文本文件并对应 writer
split_files = {
    'train': open(os.path.join(split_dir, 'ChestMinist_train.txt'), 'w', newline=''),
    'val': open(os.path.join(split_dir, 'ChestMinist_val.txt'), 'w', newline=''),
    'test': open(os.path.join(split_dir, 'ChestMinist_test.txt'), 'w', newline=''),
}
split_writers = {
    'train': csv.writer(split_files['train'], delimiter=' '),
    'val': csv.writer(split_files['val'], delimiter=' '),
    'test': csv.writer(split_files['test'], delimiter=' '),
}

# 用于统一命名所有图片（防止覆盖）
global_index = 0

# 遍历图像数据并保存
for split in ['train', 'val', 'test']:
    image_key = f"{split}_images"
    label_key = f"{split}_labels"

    image_data = npz_file[image_key]
    label_data = npz_file[label_key]

    for idx, (img_array, label) in enumerate(zip(image_data, label_data)):
        # 构造文件名（保证唯一）
        img_filename = f"{str(global_index).zfill(8)}_000.png"
        global_index += 1

        # 保存图片
        img = Image.fromarray(img_array.astype(np.uint8))
        if len(img_array.shape) == 2:
            img = img.convert('L')
        img.save(os.path.join(picture_dir, img_filename))

        # 保存标签行
        label_vector = ' '.join(map(str, label.tolist()))
        split_writers[split].writerow([img_filename] + label_vector.split())

        print(f"[{split}] Saved image: {img_filename} with label: {label_vector}")

# 关闭所有 split 文件
for f in split_files.values():
    f.close()

print("✅ All images and labels have been successfully saved across train, val, and test.")

# 把原始有文件夹层次的所有图片放入一个文件夹中
import os
import shutil
import re


def flat_training_set_img():
    from_dir = "./data/training-set-img/raw"
    to_dir = "./data/training-set-img"

    move(from_dir, to_dir)


def flat_test_set_img():
    from_dir = "./data/test-set-img/raw"
    to_dir = "./data/test-set-img"

    move(from_dir, to_dir)


def move(from_dir, to_dir):
    arr_img_path = []

    fill_arr_img_path(arr_img_path, from_dir)

    for path in arr_img_path:
        if not check_path(path):
            print(path + " 文件名有问题")
            continue

        shutil.copy(path, to_dir)


def fill_arr_img_path(arr_img_path, dir):
    for root, dirs, files in os.walk(dir):
        for name in files:
            arr_img_path.append(os.path.join(root, name))


img_filename_set = set()


def check_path(path: str):
    filename = path.split("/")[-1].split("\\")[-1]

    if filename in img_filename_set:
        print(f"{filename} 重复出现")
        return False

    if not re.match(r"^0000-\d{4}[LR]_[12]\d{3}\.jpg$", filename):
        print(f"{filename} 不匹配正则")
        return False

    img_filename_set.add(filename)
    return True


flat_test_set_img()
flat_training_set_img()

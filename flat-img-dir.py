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

    for root, dirs, files in os.walk(from_dir):
        for name in files:
            arr_img_path.append(os.path.join(root, name).replace("\\", "/"))

    for path in arr_img_path:
        if not check_path(path):
            continue

        shutil.move(path, to_dir)


img_filename_set = set()


def check_path(path: str):
    filename = path.split("/")[-1].split("\\")[-1]
    filename, ext = filename.split(".")

    if ext != "jpg":
        return False

    if filename in img_filename_set:
        print(f"{path} 重复出现")
        return False

    if not re.match(r"^0000-\d{4}[LR]_[12]", filename):
        print(f"{path} 不匹配正则")
        return False

    img_filename_set.add(filename)
    return True


flat_test_set_img()
flat_training_set_img()

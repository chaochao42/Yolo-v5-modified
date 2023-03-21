import os
import numpy as np
import cv2
import random
from pycocotools.coco import COCO

# 指定文件夹和 COCO 标注文件路径
source_name = "clear_bullet"
image_folder = os.path.join("../datasets/qf", source_name, "images")
cropped_folder = "../datasets/qf/cropped_images"
annotation_file = os.path.join("../datasets/qf", source_name, "coco_instances.json")

# 初始化 COCO API
coco = COCO(annotation_file)

# 获取所有图像的 ID
image_ids = coco.getImgIds()


images_labels_pairs = []
# 遍历所有图像 ID
for image_id in image_ids:
    # 获取图像信息和文件名
    image_info = coco.loadImgs(image_id)[0]
    image_file = os.path.join(image_folder, image_info["file_name"])

    # 读取图像并获取其尺寸
    image = cv2.imread(image_file)
    height, width, _ = image.shape

    # 获取图像的所有目标信息
    ann_ids = coco.getAnnIds(imgIds=image_id)
    anns_all = coco.loadAnns(ann_ids)

    # 遍历所有目标信息，并进行裁剪
    target_count = 0
    anns = sorted(anns_all, key=lambda x: x["category_id"], reverse=True)
    for ann in anns:
        # 获取目标的坐标信息和类别 ID
        category_id = ann["category_id"]
        label = "wo"
        if category_id == 6:
            target_count += 1
            x_1, y_1, w_1, h_1 = ann["bbox"]
            # 对目标进行裁剪
            x1_1, y1_1, x2_1, y2_1 = int(x_1), int(y_1), int(x_1+w_1), int(y_1+h_1)
        if category_id == 3 or category_id == 2:
            x, y, w, h = ann["bbox"]
            # 对目标进行裁剪
            x1, y1, x2, y2 = int(x), int(y), int(x+w), int(y+h)
            if x1 < x1_1 and x2 > x2_1 and y1 < y1_1 and y2 > y2_1:
                label = "wi"
            cropped_image = image[y1:y2, x1:x2]

            # 将裁剪后的图像保存到文件夹中，文件名为：原始文件名_目标编号_类别编号.jpg
            cropped_file_name = os.path.splitext(image_info["file_name"])[0].replace(' ', '') + "_" + source_name + str(ann["id"]) + "_" + str(category_id) + ".jpg"

            #print(os.path.join(cropped_folder, label, cropped_file_name))
            os.makedirs(os.path.join(cropped_folder, 'train', label), exist_ok=True)
            os.makedirs(os.path.join(cropped_folder, 'test', label), exist_ok=True)
            if random.random() < 0.7:
                cv2.imwrite(os.path.join(cropped_folder, 'train', label, cropped_file_name), cropped_image)
            else:
                cv2.imwrite(os.path.join(cropped_folder, 'test', label, cropped_file_name), cropped_image)
        else:
            continue

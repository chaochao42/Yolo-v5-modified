import json

# COCO标注文件路径
coco_annotations = '../datasets/qf/part_bullet/coco_instances.json'

# YOLOv5标注文件路径
yolo_annotations = '../datasets/qf/part_bullet/annotations.txt'

# COCO标签与YOLO标签的映射关系
coco2yolo = {
    'PQ': 0,
    'RH': 1,
    'LH': 2,
    'RL': 3,
    'LL': 4,
    'BD': 5
}

# 读取COCO标注文件
with open(coco_annotations, 'r') as f:
    annotations = json.load(f)

# 遍历每张图片的标注信息，转换为YOLO格式并保存
with open(yolo_annotations, 'w') as f:
    for annotation in annotations['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        bbox = annotation['bbox']

        # COCO格式的bbox为[x,y,width,height]，转换为YOLO格式的[x_center,y_center,width,height]
        x_center = bbox[0] + bbox[2] / 2
        y_center = bbox[1] + bbox[3] / 2
        width = bbox[2]
        height = bbox[3]

        # COCO类别转换为YOLO类别
        category_name = annotations['categories'][category_id - 1]['name']
        print(category_name)
        yolo_class = coco2yolo[category_name]

        # 将YOLO格式的标注信息保存到文件
        line = f"{yolo_class} {x_center} {y_center} {width} {height}\n"
        f.write(line)
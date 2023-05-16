import os
import glob
import numpy as np
import torch
import cv2
from skimage import morphology
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.torch_utils import select_device
from utils.general import check_img_size, non_max_suppression, scale_boxes, scale_segments
from utils.segment.general import process_mask, scale_image

conf_thres=0.25
iou_thres=0.45
max_det=1000

def recog_seg(file_name, weight_dir, out_dir, seg_thresh):


    device = select_device('1')

    weights = weight_dir

    imgsz = [1920, 1080]
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)

    model.warmup(imgsz=(1, 3, *imgsz))

    im0 = cv2.imread(file_name)

    im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]  # padded resize


    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(model.device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred, proto = model(im)[:2]
    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det, nm=32)

    ret = []
    for i, det in enumerate(pred):
        if len(det):
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
            #print(masks.shape, im.shape, im0.shape)


            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                defects = {'type': c,
                           'pixel': 0,
                           'area': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                           'score': 0,
                           'path': [] }
                ret.append(defects)
        else:
            return False
    colors = [(0, 0, 255) for i in range(masks.shape[0])]
    #alpha = 0.5
    alpha = 1
    colors = torch.tensor(colors, device=device, dtype=torch.float32) / 255.0  # (n, 3)
    colors = colors[:, None, None]  # shape(n,1,1,3)
    if len(masks.shape) == 3:
        masks = masks.unsqueeze(3)  # shape(n,h,w,1)

    masks_color = masks * (colors * alpha)  # shape(n,h,w,3)

    # print((1-masks * alpha).shape)
    # print((1-masks*alpha).cumprod(0).shape)
    inv_alph_masks = (1 - masks * alpha).cumprod(0)  # shape(n,h,w,1)

    mcs = (masks_color * inv_alph_masks).sum(0) * 2  # mask color summand shape(n,h,w,3)

    im_gpu = im[0]
    im_gpu = im_gpu.flip(dims=[0])  # flip channel
    im_gpu = im_gpu.permute(1, 2, 0).contiguous()  # shape(h,w,3)
    # im_gpu = im_gpu * inv_alph_masks[-1] + mcs
    #im_gpu = mcs
    im_gpu = im_gpu * inv_alph_masks[-1]
    im_mask = (masks_color.sum(0)* 255).byte().cpu().numpy()

    im0_with_mask = scale_image(im_gpu.shape, im_mask, im0.shape)

    img0_with_mask_det = np.zeros_like(im0_with_mask)

    sub_images_list = []
    distance_lis = []
    for de in ret:
       t = de['type']
       color = (0,0,255)
       tan_val = (de['area'][1]-de['area'][3])/(de['area'][0]-de['area'][2])
       # tan_val = tan_val if tan_val < 1 else 1 / tan_val
       sub_images_list.append({"img":
                                img0_with_mask_det[de['area'][1]:de['area'][3], de['area'][0]:de['area'][2] ,:],
                               "coord":
                                [de['area'][1], de['area'][3], de['area'][0], de['area'][2]],
                               "tan_box": tan_val,
                               })
       cv2.rectangle(img0_with_mask_det, (de['area'][0], de['area'][1]), (de['area'][2], de['area'][3]), color, thickness=2)
       cv2.line(img0_with_mask_det, (de['area'][0], de['area'][1]), (de['area'][2], de['area'][3]), color, thickness=2)


    for sub_image_item in sub_images_list:
        sub_image = sub_image_item["img"]
        sub_gray_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
        sub_image_det, sub_image_binary = cv2.threshold(sub_gray_image, 12, 255, cv2.THRESH_BINARY)
        sub_image_binary[sub_image_binary == 255] = 1
        sub_skeleton0 = morphology.skeletonize(sub_image_binary)  # 骨架提取
        sub_skeleton = sub_skeleton0.astype(np.uint8) * 255
        sub_image_item.update({"skeleton": sub_skeleton})

        contours, hierarchy = cv2.findContours(sub_skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        [vx, vy, x, y] = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
        rows, cols = sub_skeleton.shape[:2]
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)

        fitted_img = cv2.line(sub_skeleton, (cols - 1, righty), (0, lefty), color = 255, thickness=5)

        height, width = sub_image_binary.shape
        distances = np.zeros((height, width))
        line = np.array([[cols - 1, righty], [0, lefty]])
        # 计算每个非零像素到直线的距离
        for i in range(height):
            for j in range(width):
                if sub_image_binary[i, j] > 0:
                    distances[i, j] = cv2.pointPolygonTest(line, (j, i), True)
        max_value = max(distances.min(), distances.max(), key=abs)
        # 计算距离之和
        distance_sum = np.sum(distances) / (max_value)
        distance_lis.append(distance_sum)
        sub_image_item.update({"fitted_img": fitted_img})
        sub_image_item.update({"distance_sum": distance_sum})
        sub_image_item.update({"tan": abs((righty - lefty)/(cols - 1))})



    gray_image_det = cv2.cvtColor(img0_with_mask_det, cv2.COLOR_BGR2GRAY)
    ret_thresh_det, thresh_det = cv2.threshold(gray_image_det, 12, 255,cv2.THRESH_BINARY)

    # thresh_det[skeleton==255] = 255

    result = False
    print(distance_lis)
    mean = np.mean(distance_lis)  # 计算平均值
    std = np.std(distance_lis,ddof=0) #计算标准差
    cv = std/mean
    print(cv)
    for sub_image_item in sub_images_list:
        coords = sub_image_item["coord"]
        thresh_det[coords[0]:coords[1], coords[2]:coords[3]] = sub_image_item["fitted_img"]
        cv2.putText(img=thresh_det, text=str(sub_image_item["tan"]), org=(coords[2]+30, coords[0]+30), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1,
                    color=255, thickness=1)
        cv2.putText(img=thresh_det, text=str(sub_image_item["tan_box"])[:5], org=(coords[2]+60, coords[0]+60), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1,
                    color=255, thickness=1)
        print(sub_image_item["distance_sum"])
        # if abs(sub_image_item["tan"] - sub_image_item["tan_box"]) > seg_thresh:
        #     result = True
        #     cv2.putText(img=thresh_det, text='No', org=(coords[2], coords[0]), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1,
        #                 color=255, thickness=1)

    cv2.imwrite(os.path.join(out_dir, os.path.basename(file_name).replace(".png", "_fited_img.png")), thresh_det)

    return result
if __name__ == "__main__":
    # img_dir = "../sdfstudio/data/selfdata/normal_test_view/images/*.png"
    img_dir = "../sdfstudio/data/selfdata/normal_test_view/images/Image0058.png"

    weight_dir = "./yolov5m-seg.pt"
    out_dir = "./test_images"
    result = recog_seg(img_dir, weight_dir, out_dir, seg_thresh=0.2)
    print(result)
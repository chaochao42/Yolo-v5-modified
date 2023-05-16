import tornado.process
import tornado.web
import tornado.escape
import tornado.log
import tornado.options
import tornado.ioloop

# import argparse
import asyncio
import base64
import cv2
import numpy as np
import os
import torch
import os
import glob
import shutil
from skimage import morphology

from models.experimental import attempt_load
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.general import check_img_size, non_max_suppression, scale_boxes, scale_segments, scale_coords
from utils.segment.general import process_mask, scale_image

from utils.torch_utils import select_device

from util import check_intersection

conf_thres = 0.25
iou_thres = 0.45
max_det = 1000


def recog_seg(
        cfg,
        im0,
        seg_thres,
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        max_det=1000,
):

    imgsz = cfg.imgsz

    im = letterbox(im0, imgsz, cfg.stride)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(cfg.device)
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred, proto = cfg.model(im)[:2]

    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det, nm=32)

    ret = []
    for i, det in enumerate(pred):
        if len(det):
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)
            # print(masks.shape, im.shape, im0.shape)

            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # rescale boxes to im0 size

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                defects = {'type': c,
                           'pixel': 0,
                           'area': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                           'score': 0,
                           'path': []}
                ret.append(defects)
        else:
            return False
    colors = [(0, 0, 255) for i in range(masks.shape[0])]
    # alpha = 0.5
    alpha = 1
    colors = torch.tensor(colors, device=device, dtype=torch.float32) / 255.0
    colors = colors[:, None, None]
    if len(masks.shape) == 3:
        masks = masks.unsqueeze(3)
    masks_color = masks * (colors * alpha)
    inv_alph_masks = (1 - masks * alpha).cumprod(0)
    mcs = (masks_color * inv_alph_masks).sum(0) * 2
    im_gpu = im[0]
    im_gpu = im_gpu.flip(dims=[0])
    im_gpu = im_gpu.permute(1, 2, 0).contiguous()
    im_gpu = im_gpu * inv_alph_masks[-1]
    im_mask = (masks_color.sum(0) * 255).byte().cpu().numpy()

    im0_with_mask = scale_image(im_gpu.shape, im_mask, im0.shape)

    img0_with_mask_det = np.zeros_like(im0_with_mask)

    sub_images_list = []
    distance_list = []

    if ret == None:
        return [False, None]

    for de in ret:
        t = de['type']
        color = (0, 0, 255)
        tan_val = (de['area'][1] - de['area'][3]) / (de['area'][0] - de['area'][2])
        # tan_val = tan_val if tan_val < 1 else 1 / tan_val
        sub_images_list.append({"img":
                                    img0_with_mask_det[de['area'][1]:de['area'][3], de['area'][0]:de['area'][2], :],
                                "coord":
                                    [de['area'][1], de['area'][3], de['area'][0], de['area'][2]],
                                "tan_box": tan_val, })
        cv2.rectangle(img0_with_mask_det, (de['area'][0], de['area'][1]), (de['area'][2], de['area'][3]), color,
                      thickness=2)
        cv2.line(img0_with_mask_det, (de['area'][0], de['area'][1]), (de['area'][2], de['area'][3]), color, thickness=2)
    for sub_image_item in sub_images_list:
        sub_image = sub_image_item["img"]
        sub_gray_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2GRAY)
        sub_image_det, sub_image_binary = cv2.threshold(sub_gray_image, 12, 255, cv2.THRESH_BINARY)
        sub_image_binary[sub_image_binary == 255] = 1
        sub_skeleton0 = morphology.skeletonize(sub_image_binary)
        sub_skeleton = sub_skeleton0.astype(np.uint8) * 255
        sub_image_item.update({"skeleton": sub_skeleton})
        contours, hierarchy = cv2.findContours(sub_skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        [vx, vy, x, y] = cv2.fitLine(contours[0], cv2.DIST_L2, 0, 0.01, 0.01)
        rows, cols = sub_skeleton.shape[:2]
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)

        fitted_img = cv2.line(sub_skeleton, (cols - 1, righty), (0, lefty), color=255, thickness=5)
        height, width = sub_image_binary.shape
        distances = np.zeros((height, width))
        line = np.array([[cols - 1, righty], [0, lefty]])
        for i in range(height):
            for j in range(width):
                if sub_image_binary[i, j] > 0:
                    distances[i, j] = cv2.pointPolygonTest(line, (j, i), True)

        max_value = max(distances.min(), distances.max(), key=abs)
        distance_sum = np.sum(distances) / (max_value)  # * np.sum(sub_image_binary))
        # distance_sum = np.sum(distances) #/ np.sum(sub_image_binary)
        distance_list.append(distance_sum)

        sub_image_item.update({"distance_sum": abs(distance_sum)})
        sub_image_item.update({"fitted_img": fitted_img})
        sub_image_item.update({"tan": abs((righty - lefty) / (cols - 1))})

    sub_images_list = sorted(sub_images_list, key=lambda e: e['distance_sum'], reverse=True)
    gray_image_det = cv2.cvtColor(img0_with_mask_det, cv2.COLOR_BGR2GRAY)
    ret_thresh_det, thresh_det = cv2.threshold(gray_image_det, 12, 255, cv2.THRESH_BINARY)

    mean = np.mean(distance_list)  #
    std = np.std(distance_list, ddof=0)  #
    cv = std / mean
    result = cv > seg_thres
    result_lis = []
    result_lis.append = [result, sub_images_list[0]]
    return result_lis


def spreader_seg(
        cfg,
        im0,
        seg_thres,
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        max_det=1000,
):

    imgsz = cfg.imgsz

    im = letterbox(im0, imgsz, cfg.stride)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(model.device)
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred, proto = cfg.model(im)[:2]

    pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=max_det, nm=32)

    for i, det in enumerate(pred):
        if len(det):
            masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)


    colors = [(0, 0, 255) for i in range(masks.shape[0])]
    # alpha = 0.5
    alpha = 1
    colors = torch.tensor(colors, device=device, dtype=torch.float32) / 255.0
    colors = colors[:, None, None]
    if len(masks.shape) == 3:
        masks = masks.unsqueeze(3)
    masks_color = masks * (colors * alpha)
    inv_alph_masks = (1 - masks * alpha).cumprod(0)
    mcs = (masks_color * inv_alph_masks).sum(0) * 2
    im_gpu = im[0]
    im_gpu = im_gpu.flip(dims=[0])
    im_gpu = im_gpu.permute(1, 2, 0).contiguous()
    im_gpu = im_gpu * inv_alph_masks[-1] + mcs
    im_mask = (masks_color.sum(0) * 255).byte().cpu().numpy()

    im0_with_mask = scale_image(im_gpu.shape, im_mask, im0.shape)

    im0[:] = im0_with_mask


    return im0

# yolov5 model class
class YoloDetect():
    def __init__(self) -> None:
        self.model = None
        self.device = None
        self.names = None
        self.stride = None
        self.imgsz = None


# yolo detect method
@torch.no_grad()
def detect(cfg,
           img0,
           conf_thres,  # confidence threshold
           iou_thres=0.5,  # NMS IOU threshold
           max_det=1000,  # maximum detections per image
           classes=None,  # filter by class: --class 0, or --class 0 2 3
           agnostic_nms=False,  # class-agnostic NMS
           augment=False,  # augmented inference
           ):
    ret = []

    if cfg.model is None:
        return ret

    # Set Dataloader
    img = letterbox(img0, cfg.imgsz, stride=cfg.stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Run inference
    img = torch.from_numpy(img).to(cfg.device)
    # img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = cfg.model(img, augment=augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                defects = {'type': c,
                           'pixel': 0,
                           'area': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                           'score': 0,
                           'path': []}
                ret.append(defects)

    return ret


# yolo load model method
def loadModel(weights, imgsz):
    if weights is None or weights == '' or not os.path.exists(weights):
        return None

    cfg = YoloDetect()

    cfg.device = select_device('')
    # Load model
    cfg.model = attempt_load(weights, cfg.device)  # load FP32 model
    cfg.stride = int(cfg.model.stride.max())  # model stride
    cfg.imgsz = check_img_size(imgsz, s=cfg.stride)  # check image size
    cfg.names = cfg.model.module.names if hasattr(cfg.model, 'module') else cfg.model.names  # get class names

    cfg.model(torch.zeros(1, 3, cfg.imgsz, cfg.imgsz).to(cfg.device).type_as(next(cfg.model.parameters())))  # run once
    return cfg


# Web setting class
class WebSettings:
    # default http port
    DEFAULT_PORT = 8888
    # default model file path
    DEFAULT_MODEL = "model.pt"
    # default para file path
    DEFAULT_PARA = "para.json"
    # default stat file path
    DEFAULT_STAT = "stat.csv"
    # default stat enable
    DEFAULT_STAT_ENABLE = True
    # default threshold
    DEFAULT_THRE = 0.1
    # default image size
    DEFAULT_IMGSZ = 640
    # default whether vis open
    DEFAULT_VIS_OPEN = True
    # default whether vis sling
    DEFAULT_VIS_SLING = True
    # defect names
    DEFECT_NAMES = ['DOOR', 'SLING', 'HB', 'GRADE2', 'GRADE3', 'OPEN']
    # color table
    COLOR_TABLE = {
        "RED": (0, 0, 255),
        "GREEN": (0, 255, 0),
        "BLUE": (255, 0, 0),
        "YELLOW": (0, 255, 255),
        "CYAN": (0, 128, 128),
        "ORANGE": (0, 128, 255),
        "BLACK": (0, 0, 0),
    }
    STAT_INTERVAL = 60

    def __init__(self) -> None:
        self._model1 = None
        self._model2 = None
        self._model3 = None
        self._model_open = None
        self._model_aux = None
        self._model_seg = None
        self._model_seg_spreader = None
        self._para = None
        self._parafile = WebSettings.DEFAULT_PARA
        self._colors = []
        self._statfile = WebSettings.DEFAULT_STAT
        self._statenable = WebSettings.DEFAULT_STAT_ENABLE
        self._end_time = time.time() + WebSettings.STAT_INTERVAL
        self._count = 0
        self._total_time = 0

    # load yolo model
    def loadModel(self, model1, model2, model3, model_open, model_aux, model_seg, model_seg_spreader, thre1, thre2, thre3, thre_open,
                  thre_aux, thre_seg, thre_seg_spreader, imgsz):
        self._model1 = loadModel(model1, imgsz)
        self._model2 = loadModel(model2, imgsz)
        self._model3 = loadModel(model3, imgsz)
        self._model_open = loadModel(model_open, imgsz)
        self._model_aux = loadModel(model_aux, imgsz)
        self._model_seg = loadModel(model_seg, imgsz)
        self._model_seg_spreader = loadModel(model_seg_spreader, imgsz)
        self._thre1 = thre1
        self._thre2 = thre2
        self._thre3 = thre3
        self._thre_open = thre_open
        self._thre_aux = thre_aux
        self._thre_seg = thre_seg
        self._thre_seg_spreader = thre_seg_spreader
        if self._model1 and self._model2 and self._model3 and self._model_open and self._model_aux and self._model_seg and self._model_seg_spreader:
            return True
        return False

    # load stat
    def loadStat(self, enable, statfile):
        self._statenable = enable
        if not self._statenable:
            return
        self._statfile = statfile
        if os.path.isfile(self._statfile):
            return
        with open(self._statfile, 'w', newline='', encoding='utf-8') as f:
            self._statf = csv.writer(f)
            self._statf.writerow(['时间戳', '图片推理数量', '图片平均推理时间', '最大请求排队数量'])

    # put stat
    def putStat(self, elapsed):
        if not self._statenable:
            return
        self._count += 1
        self._total_time += elapsed

    # write stat
    def writeStat(self):
        if not self._statenable:
            return
        cur = time.time()
        if cur < self._end_time:
            return
        average_time = self._total_time / self._count if self._count > 0 else 0
        timestamp_str = datetime.datetime.fromtimestamp(cur).strftime('%Y-%m-%d %H:%M:%S')
        num = len([task for task in asyncio.all_tasks() if isinstance(task, asyncio.Future) and not task.done()])
        with open(self._statfile, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp_str, self._count, average_time, num])
        self._end_time = cur + WebSettings.STAT_INTERVAL
        self._count = 0
        self._total_time = 0

    # load para
    def loadPara(self, parafile):
        self._parafile = parafile
        if not os.path.exists(self._parafile):
            return False
        with open(self._parafile, "r", encoding='utf-8') as f:
            self.para = tornado.escape.json_decode(f.read())
        print(f"para: {self.para}")
        return True

    # load my para
    def loadMyPara(self, vis_sling, vis_door):
        self.my_para = {"ifVisSling": vis_sling, "ifVisDoor": vis_door}
        print(f"my_para: {self.my_para}")
        return True

    @property
    def model1(self):
        return self._model1

    @property
    def model2(self):
        return self._model2

    @property
    def model3(self):
        return self._model3

    @property
    def model_open(self):
        return self._model_open

    @property
    def model_aux(self):
        return self._model_aux

    @property
    def model_seg(self):
        return self._model_seg

    @property
    def model_seg_spreader(self):
        return self._model_seg_spreader
    @property
    def para(self):
        return self._para

    @para.setter
    def para(self, p):
        self._para = p
        self._colors = []
        colors = self.getParam("labelColor")
        if colors is None:
            return
        self._colors = [x.strip().upper() for x in colors.split(',')]

    # get para
    def getParam(self, name):
        return self._para[name] if name in self._para else None

    # get para
    def getMyParam(self, name):
        return self.my_para[name] if name in self.my_para else None

    # save para
    def savePara(self, para):
        self.para = para
        with open(self._parafile, "w", encoding='utf-8') as f:
            f.write(tornado.escape.json_encode(self._para))

    @property
    def isTypeLabel(self):
        f = self.getParam('isTypeLabel')
        return False if f is None else f

    @property
    def isPosLabel(self):
        f = self.getParam('isPosLabel')
        return True if f is None else f

    @property
    def ifVisSling(self):
        f = self.getMyParam('ifVisSling')
        return True if f is None else f

    @property
    def ifVisDoor(self):
        f = self.getMyParam('ifVisDoor')
        return True if f is None else f

    # get name and color from defect type
    def getNameAndColor(self, type):
        name = WebSettings.DEFECT_NAMES[type]
        if len(self._colors) <= 0:
            return "", WebSettings.COLOR_TABLE["RED"]
        if type >= len(self._colors):
            type = 0
        return name, WebSettings.COLOR_TABLE[self._colors[type]]


# /ocrdamage handler
class InferenceHandler(tornado.web.RequestHandler):

    def initialize(self, settings):
        self._websettings: WebSettings = settings

    def get(self):
        self.write("Hello, world")

    def post(self):
        start = time.time()
        # get request arguments
        fn = self.get_body_argument("fileName")
        sz = self.get_body_argument("size")
        cv = self.get_body_argument("ctView")
        tornado.log.app_log.info(f"{fn} {sz} {cv}")

        # get request image data and convert to cv image
        file = self.request.files['pics'][0]
        imgdata = np.asarray(bytearray(file.body), dtype="uint8")
        img = cv2.imdecode(imgdata, cv2.IMREAD_COLOR)

        # inference all defects
        defects1 = detect(self._websettings.model1, img, self._websettings._thre1)
        defects2 = detect(self._websettings.model2, img, self._websettings._thre2)
        defects3 = detect(self._websettings.model3, img, self._websettings._thre3)
        defects_open = detect(self._websettings.model_open, img, self._websettings._thre_open)
        door_back_sling = detect(self._websettings.model_aux, img, self._websettings._thre_aux)
        print("===================================================")
        # seg-related
        door_pillor_blending = recog_seg(self._websettings.model_seg, img, self._websettings._thre_seg)
        ## draw spreader
        img = spreader_seg(self._websettings.model_seg_spreader, img, self._websettings._thre_seg_spreader)
        # door & sling
        print("++++++++++++++++++++++++++++++++++++++++++++++++++")
        bboxes_door = []
        bboxes_sling = []
        for bbox in door_back_sling:
            # 0: door, 1: back, 2: sling
            if bbox["type"] == 0:
                name, color = self._websettings.getNameAndColor(0)
                bboxes_door.append(bbox['area'])
                # draw sling rectangle
                if self._websettings.ifVisDoor and self._websettings.isPosLabel:
                    cv2.rectangle(img, (bbox['area'][0], bbox['area'][1]), (bbox['area'][2], bbox['area'][3]), color,
                                  thickness=1)
                # draw sling name
                if self._websettings.ifVisDoor and self._websettings.isTypeLabel:
                    cv2.putText(img, name, (bbox['area'][0], bbox['area'][1]), 0, 0.75, color, thickness=2)
            elif bbox["type"] == 2:
                name, color = self._websettings.getNameAndColor(1)
                bboxes_sling.append(bbox['area'])
                # draw sling rectangle
                if self._websettings.ifVisSling and self._websettings.isPosLabel:
                    cv2.rectangle(img, (bbox['area'][0], bbox['area'][1]), (bbox['area'][2], bbox['area'][3]), color,
                                  thickness=1)
                # draw sling name
                if self._websettings.ifVisSling and self._websettings.isTypeLabel:
                    cv2.putText(img, name, (bbox['area'][0], bbox['area'][1]), 0, 0.75, color, thickness=2)

        # defects
        delist = []
        # door blending
        if door_pillor_blending[0] == True:
            delist.append({"defectType": "door_pillor_blending"})

        for idx, defects in enumerate([defects1, defects2, defects3, defects_open]):
            for de in defects:
                # filter out sling
                if check_intersection(de['area'], bboxes_sling, iob_thre=0.5):
                    continue
                # open only in door
                if idx == 3 and not check_intersection(de['area'], bboxes_door, iob_thre=0.1):
                    continue

                t = de["type"] + idx + 2
                name, color = self._websettings.getNameAndColor(t)

                # draw defect rectangle
                if self._websettings.isPosLabel:
                    cv2.rectangle(img, (de['area'][0], de['area'][1]), (de['area'][2], de['area'][3]), color,
                                  thickness=1)

                # draw defect name
                if self._websettings.isTypeLabel:
                    cv2.putText(img, name, (de['area'][0], de['area'][1]), 0, 0.75, color, thickness=2)

                delist.append({"defectType": name,
                               "points": f"{de['area'][0]:.2f},{de['area'][1]:.2f},{de['area'][2]:.2f},{de['area'][3]:.2f}"})
        # convert image to base64 data
        ret, imgdata = cv2.imencode(".jpg", img)
        img64 = base64.b64encode(imgdata).decode()

        # response
        jo = {
            "resulType": 0,
            "errorMessage": "",
            "data": {"data": img64, "defects": delist}
        }
        self.set_header("Content-Type", "application/json")
        self.write(tornado.escape.json_encode(jo))
        end = time.time()
        self._websettings.putStat(end - start)

# /paraconfig handler
class ParamConfigHandler(tornado.web.RequestHandler):

    def initialize(self, settings):
        self._websettings: WebSettings = settings

    def get(self):
        self.write("Hello, world")

    def post(self):
        # get request json data
        json = self.request.body.decode('utf-8')
        request = tornado.escape.json_decode(json)
        tornado.log.app_log.info(request)

        n = request['parameters']
        del request['parameters']
        m = len(request.keys())

        if m != n:
            jo = {
                "resulType": 1000,
                "errorMessage": f"Error: Parameter number mismatch, expected {n} para while actual input {m}",
                "data": False
            }
            self.set_header("Content-Type", "application/json")
            self.write(tornado.escape.json_encode(jo))
            return

        for key in request.keys():
            v = self._websettings.getParam(key)
            if v is None:
                jo = {
                    "resulType": 1010,
                    "errorMessage": f"Error：Parameter ‘{key}’ does not exist",
                    "data": False
                }
                self.set_header("Content-Type", "application/json")
                self.write(tornado.escape.json_encode(jo))
                return
            if not isinstance(request[key], type(v)):
                jo = {
                    "resulType": 1020,
                    "errorMessage": f"Error: Invalid parameter value, {key}",
                    "data": False
                }
                self.set_header("Content-Type", "application/json")
                self.write(tornado.escape.json_encode(jo))
                return

                # save para
        self._websettings.savePara(request)

        # response
        jo = {
            "resulType": 0,
            "errorMessage": "",
            "data": True
        }
        self.set_header("Content-Type", "application/json")
        self.write(tornado.escape.json_encode(jo))


# /paralist handler
class ParamListHandler(tornado.web.RequestHandler):

    def initialize(self, settings):
        self._websettings: WebSettings = settings

    def get(self):
        # response para list
        jo = {
            "resulType": 0,
            "errorMessage": "",
            "data": self._websettings.para
        }
        self.set_header("Content-Type", "application/json")
        self.write(tornado.escape.json_encode(jo))

async def stat(websettings):
    while True:
        websettings.writeStat()
        await asyncio.sleep(1)
# create tornado web app
def make_app():
    settings = WebSettings()
    settings.loadStat(tornado.options.options.pf_monitor, tornado.options.options.stat)

    if tornado.options.options.pf_monitor:
        asyncio.ensure_future(stat(settings))
    # load model
    tornado.log.app_log.info("load model")
    if not settings.loadModel( \
            tornado.options.options.model1, \
            tornado.options.options.model2, \
            tornado.options.options.model3, \
            tornado.options.options.model_open, \
            tornado.options.options.model_aux, \
            tornado.options.options.model_seg,\
            tornado.options.options.model_seg_spreader,\
            tornado.options.options.thre1, \
            tornado.options.options.thre2, \
            tornado.options.options.thre3, \
            tornado.options.options.thre_open, \
            tornado.options.options.thre_aux, \
            tornado.options.options.thre_seg,
            tornado.options.options.thre_seg_spreader,
            tornado.options.options.imgsz, \
            ):
        tornado.log.app_log.info(f"load model failed, web app exit")
        exit()

    tornado.log.app_log.info("load para")
    if not settings.loadPara(tornado.options.options.para):
        tornado.log.app_log.info(
            f"load para {tornado.options.options.para} failed, remember to request /paraconfig first! ")

    tornado.log.app_log.info("load my_para")
    if not settings.loadMyPara(tornado.options.options.vis_sling, tornado.options.options.vis_door):
        tornado.log.app_log.info(f"load my_para failed! ")

    # register web app route
    return tornado.web.Application([
        (r"/ocrdamage", InferenceHandler, dict(settings=settings)),
        (r"/paraconfig", ParamConfigHandler, dict(settings=settings)),
        (r"/paralist", ParamListHandler, dict(settings=settings)),
        # static file for self-test
        (r"/(.*)", tornado.web.StaticFileHandler, {"path": "./www/", "default_filename": "index.html"}),
    ])


# main
async def main():
    app = make_app()
    app.listen(tornado.options.options.port)
    shutdown_event = asyncio.Event()
    tornado.log.app_log.info("Ready")
    await shutdown_event.wait()


if __name__ == "__main__":
    # arguments parse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model1', type=str, default=WebSettings.DEFAULT_MODEL, help=f'model file path, default is [{WebSettings.DEFAULT_MODEL}]')
    # parser.add_argument('--model2', type=str, default=WebSettings.DEFAULT_MODEL, help=f'model file path, default is [{WebSettings.DEFAULT_MODEL}]')
    # parser.add_argument('--model3', type=str, default=WebSettings.DEFAULT_MODEL, help=f'model file path, default is [{WebSettings.DEFAULT_MODEL}]')
    # parser.add_argument('--thre1', type=float, default=WebSettings.DEFAULT_THRE, help=f'threshold, default is [{WebSettings.DEFAULT_THRE}]')
    # parser.add_argument('--thre2', type=float, default=WebSettings.DEFAULT_THRE, help=f'threshold, default is [{WebSettings.DEFAULT_THRE}]')
    # parser.add_argument('--thre3', type=float, default=WebSettings.DEFAULT_THRE, help=f'threshold, default is [{WebSettings.DEFAULT_THRE}]')
    # parser.add_argument('--imgsz', type=int, default=WebSettings.DEFAULT_IMGSZ, help=f'image size, default is [{WebSettings.DEFAULT_IMGSZ}]')
    # parser.add_argument('--port', type=int, default=WebSettings.DEFAULT_PORT, help=f'listen port, default is [{WebSettings.DEFAULT_PORT}]')
    # parser.add_argument('--para', type=str, default=WebSettings.DEFAULT_PARA, help=f'para file path, default is [{WebSettings.DEFAULT_PARA}]')
    # opt = parser.parse_args()

    tornado.options.define('model1', type=str, default=WebSettings.DEFAULT_MODEL, help=f'model file path')
    tornado.options.define('model2', type=str, default=WebSettings.DEFAULT_MODEL, help=f'model file path')
    tornado.options.define('model3', type=str, default=WebSettings.DEFAULT_MODEL, help=f'model file path')
    tornado.options.define('model_open', type=str, default=WebSettings.DEFAULT_MODEL, help=f'model file path')
    tornado.options.define('model_aux', type=str, default=WebSettings.DEFAULT_MODEL, help=f'model file path')
    tornado.options.define('model_seg', type=str, default=WebSettings.DEFAULT_MODEL, help=f'model file path')
    tornado.options.define('model_seg_spreader', type=str, default=WebSettings.DEFAULT_MODEL, help=f'model file path')
    tornado.options.define('thre1', type=float, default=WebSettings.DEFAULT_THRE, help=f'threshold')
    tornado.options.define('thre2', type=float, default=WebSettings.DEFAULT_THRE, help=f'threshold')
    tornado.options.define('thre3', type=float, default=WebSettings.DEFAULT_THRE, help=f'threshold')
    tornado.options.define('thre_open', type=float, default=WebSettings.DEFAULT_THRE, help=f'threshold')
    tornado.options.define('thre_aux', type=float, default=WebSettings.DEFAULT_THRE, help=f'threshold')
    tornado.options.define('thre_seg', type=float, default=WebSettings.DEFAULT_THRE, help=f'threshold')
    tornado.options.define('thre_seg_spreader', type=float, default=WebSettings.DEFAULT_THRE, help=f'threshold')
    tornado.options.define('imgsz', type=int, default=WebSettings.DEFAULT_IMGSZ, help=f'image size')
    tornado.options.define('vis_door', type=bool, default=WebSettings.DEFAULT_VIS_OPEN, help=f'whether vis open')
    tornado.options.define('vis_sling', type=bool, default=WebSettings.DEFAULT_VIS_SLING, help=f'whether vis sling')
    tornado.options.define('port', type=int, default=WebSettings.DEFAULT_PORT, help=f'listen port')
    tornado.options.define('para', type=str, default=WebSettings.DEFAULT_PARA, help=f'para file path')
    tornado.options.define('stat', type=str, default=WebSettings.DEFAULT_STAT, help=f'stat file path')
    tornado.options.define('pf_monitor', type=bool, default=WebSettings.DEFAULT_STAT_ENABLE, help=f'pf monitor')
    tornado.options.parse_command_line()

    tornado.log.app_log.info(f'model1: {tornado.options.options.model1}, thre1: {tornado.options.options.thre1}')
    tornado.log.app_log.info(f'model2: {tornado.options.options.model2}, thre2: {tornado.options.options.thre2}')
    tornado.log.app_log.info(f'model3: {tornado.options.options.model3}, thre3: {tornado.options.options.thre3}')
    tornado.log.app_log.info(
        f'model_open: {tornado.options.options.model_open}, thre_open: {tornado.options.options.thre_open}')
    tornado.log.app_log.info(
        f'model_aux: {tornado.options.options.model_aux}, thre_aux: {tornado.options.options.thre_aux}')
    tornado.log.app_log.info(
        f'model_seg: {tornado.options.options.model_seg}, thre_seg: {tornado.options.options.thre_seg}')
    tornado.log.app_log.info(
        f'model_seg_spreader: {tornado.options.options.model_seg_spreader}, thre_seg_spreader: {tornado.options.options.thre_seg_spreader}')
    tornado.log.app_log.info(f'port: {tornado.options.options.port}, para: {tornado.options.options.para}')

    asyncio.run(main())

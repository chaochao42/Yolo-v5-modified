import warnings
warnings.filterwarnings("ignore")

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
import torch.nn.functional as F
import torch


from models.common import DetectMultiBackend
from models.experimental import attempt_load
from utils.augmentations import letterbox
from utils.augmentations import classify_transforms
from utils.general import check_img_size,  non_max_suppression, scale_coords
from utils.torch_utils import select_device
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
#yolov5 model class
class YoloDetect():
    def __init__(self) -> None:
        self.model = None
        self.device = None
        self.names = None
        self.stride = None
        self.imgsz = None
        self.cls_model = None
        self.cls_imgsz = None

#yolo detect method
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
    ret_ls = []
    if cfg.model is None:
        return ret

    # Set Dataloader
    img = letterbox(img0, cfg.imgsz, stride=cfg.stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    # Run inference
    img = torch.from_numpy(img).to(cfg.device)
    #img = img.half() if half else img.float()  # uint8 to fp16/32
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # img shape torch.Size([1, 3, 384, 640]) and img0 shape (1080, 1920, 3)

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
                    if c == 0 or c == 2 or c == 3 or c == 4:

                        defects = { 'type': c,
                                    'pixel': 0,
                                    'area': [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])],
                                    'score': 0,
                                    'path': [] }
                        ret.append(defects)

    return ret


@torch.no_grad()
def classification(cfg,
                   img0,
                   conf,
                   de):


    img_in = img0[de['area'][1]:de['area'][3], de['area'][0]:de['area'][2] , :]


    transforms = classify_transforms(cfg.cls_imgsz)
    img = transforms(img_in)

    img = torch.Tensor(img).to(cfg.device)

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    result = cfg.cls_model(img)

    # pred = F.softmax(result, dim=1)
    _, preds = torch.max(result, 1)



    return preds.cpu().numpy().tolist()[0]

#yolo load model method
def loadModel(weights, cls_weights, imgsz, cls_imgsz):
    if weights is None or weights == '' or not os.path.exists(weights) or not os.path.exists(cls_weights):
        return None

    cfg = YoloDetect()

    cfg.device = select_device('')
    # Load model

    cfg.model = attempt_load(weights, cfg.device)  # load FP32 model
    cfg.cls_model = DetectMultiBackend(cls_weights, cfg.device)
    bs = 1
    stride, names, pt = cfg.cls_model.stride, cfg.cls_model.names, cfg.cls_model.pt
    cfg.cls_model.warmup(imgsz=(bs, 3, cls_imgsz, cls_imgsz))
    cfg.stride = int(cfg.model.stride.max())  # model stride
    cfg.imgsz = check_img_size(imgsz, s=cfg.stride)  # check image size
    cfg.cls_imgsz = check_img_size(cls_imgsz, s=cfg.stride)
    cfg.names = cfg.model.module.names if hasattr(cfg.model, 'module') else cfg.model.names  # get class names

    cfg.model(torch.zeros(1, 3, cfg.imgsz, cfg.imgsz).to(cfg.device).type_as(next(cfg.model.parameters())))  # run once
    return cfg


#Web setting class
class WebSettings:
    #default http port
    DEFAULT_PORT = 8880
    #default model file path
    DEFAULT_MODEL = "deploy_ckpts/best.pt"
    DEFAULT_CLS_MODEL = "deploy_ckpts/best_cls.pt"
    #defaul threshold
    DEFAULT_THRE = 0.3
    #defaul image size
    DEFAULT_IMGSZ = 640
    DEFAULT_CLS_IMGSZ = 224
    def __init__(self) -> None:
        self._model = None
        self._conf = WebSettings.DEFAULT_THRE

    #load yolo model
    def loadModel(self, model, cls_model, imgsz, cls_imgsz):
        self._model = loadModel(model, cls_model, imgsz, cls_imgsz)
        if self._model:
            return True
        return False

    @property
    def model(self):
        return self._model

    @property
    def conf(self):
        return self._conf

#/ocrdamage handler
class InferenceHandler(tornado.web.RequestHandler):

    def initialize(self, settings):
        self._websettings : WebSettings = settings

    def get(self):
        self.write("Hello, world")
    
    def post(self):
        thres = self._websettings.conf

        #get request arguments
        conf = self.get_body_argument("conf")
        tornado.log.app_log.info(f"conf: {conf}")
        if len(conf) == 0:
            conf = thres
        conf = min(1.0, max(0.0, float(conf)))
        tornado.log.app_log.info(f"conf: {conf}")

        #get request image data and convert to cv image
        file = self.request.files['pics'][0]
        imgdata = np.asarray(bytearray(file.body), dtype="uint8")
        img = cv2.imdecode(imgdata, cv2.IMREAD_COLOR)
        
        #inference all defects
        defects = detect(self._websettings.model, img, conf)

        delist = []
        color_list = [(0,255,0), (0, 0, 255)]
        label_list = ["Yes", "No"]
        isSeal_flag = False
        for de in defects:
            cls_label = classification(self._websettings.model, img, conf, de)
            t = de["type"]
            color = (255,0,0)

            if cls_label != None:
                if cls_label == 0:
                    isSeal_flag = True
                cv2.rectangle(img, (de['area'][0], de['area'][1]), (de['area'][2], de['area'][3]), color_list[cls_label], thickness=1)
                cv2.putText(img, label_list[cls_label], (int((de['area'][0]+de['area'][2])/2), int((de['area'][1]+de['area'][3])/2)), cv2.FONT_HERSHEY_COMPLEX, 0.7, color_list[cls_label], thickness=2)
            else:
                print(img.shape)
                print((de['area'][0], de['area'][1]))
                print((de['area'][2], de['area'][3]))
                cv2.rectangle(img, (de['area'][0], de['area'][1]), (de['area'][2], de['area'][3]), color, thickness=1)
            delist.append({"defectType": t, "points": f"{de['area'][0]:.2f},{de['area'][1]:.2f},{de['area'][2]:.2f},{de['area'][3]:.2f}"})

        #convert image to base64 data
        ret, imgdata = cv2.imencode(".jpg", img)
        img64 = base64.b64encode(imgdata).decode()
        
        #response
        jo = { 
            "resulType": 0,
            "errorMessage": "",
            "sealImageBase64": img64, 
            "isSeal": 1 if isSeal_flag else 0,
            "confDet": conf
        }
        self.set_header("Content-Type", "application/json")
        self.write(tornado.escape.json_encode(jo))


#create tornado web app
def make_app():
    settings = WebSettings()

    #load model
    tornado.log.app_log.info("load model")
    if not settings.loadModel(
                            tornado.options.options.model,
                            tornado.options.options.cls_model,
                            tornado.options.options.imgsz,
                            tornado.options.options.cls_imgsz,
    ):
        tornado.log.app_log.info(f"load model failed, web app exit")
        exit()

    #register web app route
    return tornado.web.Application([
        (r"/ls", InferenceHandler, dict(settings = settings)),
        #static file for self-test
        (r"/(.*)", tornado.web.StaticFileHandler, {"path": "./www/", "default_filename": "index_ls.html"}),
    ])


#main
async def main():
    app = make_app()
    app.listen(tornado.options.options.port)
    shutdown_event = asyncio.Event()
    tornado.log.app_log.info("Ready")
    await shutdown_event.wait()


if __name__ == "__main__":
    tornado.options.define('model', type=str, default=WebSettings.DEFAULT_MODEL, help=f'model file path')
    tornado.options.define('imgsz', type=int, default=WebSettings.DEFAULT_IMGSZ, help=f'image size')
    tornado.options.define('cls_model', type=str, default=WebSettings.DEFAULT_CLS_MODEL, help=f'model file path')
    tornado.options.define('cls_imgsz', type=int, default=WebSettings.DEFAULT_CLS_IMGSZ, help=f'image size')
    tornado.options.define('port', type=int, default=WebSettings.DEFAULT_PORT, help=f'listen port')
    tornado.options.parse_command_line()
    
    tornado.log.app_log.info(f'port: {tornado.options.options.port}, model: {tornado.options.options.model}, imgsz: {tornado.options.options.imgsz}')

    asyncio.run(main())


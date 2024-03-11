import os
import pdb
from typing import List, Optional, Union

import numpy as np
import torch

from norfair import Detection


class YOLO:
    def __init__(self, model_path: str, device: Optional[str] = None):
        if device is not None and "cuda" in device and not torch.cuda.is_available():
            raise Exception(
                "Selected device='cuda', but cuda is not available to Pytorch."
            )
        # automatically set device if its None
        elif device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if not os.path.exists(model_path):
            os.system(
                f"wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/{os.path.basename(model_path)} -O {model_path}"
            )

        # load model
        # pdb.set_trace()
        try:
            self.model = torch.hub.load("WongKinYiu/yolov7", "custom", model_path)
        except:
            raise Exception("Failed to load model from {}".format(model_path))

    def __call__(
        self,
        img: Union[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 720,
        classes: Optional[List[int]] = None,
    ) -> torch.tensor:

        self.model.conf = conf_threshold
        self.model.iou = iou_threshold
        if classes is not None:
            self.model.classes = classes
        detections = self.model(img, size=image_size)
        return detections


def yolo_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "centroid":
        detections_as_xywh = yolo_detections.xywh[0]
        for detection_as_xywh in detections_as_xywh:
            centroid = np.array(
                [
                    [detection_as_xywh[0].item(), detection_as_xywh[1].item()],
                    [detection_as_xywh[0].item(), detection_as_xywh[1].item()],
                ]
            )
            scores = np.array(
                [detection_as_xywh[4].item(), detection_as_xywh[4].item()]
            )
            norfair_detections.append(Detection(points=centroid, scores=scores))
    elif track_points == "bbox":
        detections_as_xyxy = yolo_detections[0].boxes.xyxy[0].cpu().detach().numpy()
        # detections_as_xyxy.cpu().detach().numpy()
        for detection_as_xyxy in detections_as_xyxy:
            bbox = np.array(
                [
                    [detection_as_xyxy[0].item(), detection_as_xyxy[1].item()],
                    [detection_as_xyxy[2].item(), detection_as_xyxy[3].item()],
                ]
            )
            scores = np.array(
                [detection_as_xyxy[4].item(), detection_as_xyxy[4].item()]
            )
            norfair_detections.append(Detection(points=bbox, scores=scores))

    return norfair_detections


def yolo_ultralytics_detections_to_norfair_detections(
    yolo_detections: torch.tensor, track_points: str = "centroid"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []

    if track_points == "bbox":
        detections_as_xyxy = yolo_detections[0].boxes.xyxy
        detections_as_conf = yolo_detections[0].boxes.conf
        detections_as_cls = yolo_detections[0].boxes.cls
        cls_names = yolo_detections[0].names
        # pdb.set_trace()

        # detections_as_xyxy.cpu().detach().numpy()
        for box, conf, cls in zip(detections_as_xyxy, detections_as_conf, detections_as_cls):
            if conf.item() < 0.5:
                continue
            bbox = np.array(
                [
                    [box[0].item(), box[1].item()],
                    [box[2].item(), box[3].item()],
                ]
            )
            scores = np.array(
                [conf.item(), conf.item()]
            )
            # print(scores)
            norfair_detections.append(Detection(points=bbox, scores=scores, label=cls_names[int(cls)]))

    return norfair_detections


def tracker_to_input(
    yolo_detections: torch.tensor, track_points: str = "bbox"  # bbox or centroid
) -> List[Detection]:
    """convert detections_as_xywh to norfair detections"""
    norfair_detections: List[Detection] = []
    # pdb.set_trace()
    for one_object in yolo_detections:

        points = one_object.estimate
        scores = np.array(
            [0.99, 0.99]
        )
        print("tracker:",scores)
        norfair_detections.append(Detection(points=points, scores=scores))

    return norfair_detections
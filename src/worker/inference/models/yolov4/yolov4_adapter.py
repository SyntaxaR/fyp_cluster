"""
An example adapter based on YOLOv4 ONNX model
You may refer to this implementation for adapting a new custom model
Template: src/worker/inference/model_adapter_template.py
Edited from: https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/yolov4
inputs shapes: (1, 416, 416, 3)
Each dimension represents the following variables: (batch_size, height, width, channels).
Output shape: (1, 52, 52, 3, 85)
There are 3 output layers. For each layer, there are 255 outputs: 85 values per anchor, times 3 anchors.
By default the postprocessing won't save the image, you may want to manually set `save_images=True` in the meta data.
"""

import numpy as np
from typing import Any, Optional, Literal
from dataclasses import dataclass
from scipy import special
import colorsys
import random
import os
from PIL import Image

import cv2


# Raw Item mode schema, only need to implement the types needed for your model
RawItemType = Literal["image_bytes", "image_path", "text"]

@dataclass
class RawItem:
    type: RawItemType
    data: Any
    mime: Optional[str] = None

class ModelAdapter:
    """
        Implement these methods to fit a custom model.
        - Required for Raw Item mode inference
            preprocess: Raw items -> tensor feed
        - Required for Dummy Input mode inference
            generate_dummy_inputs: for compute-only benchmarking
        - Required if any postprocess is needed (e.g. draw boxes and save for object detection models)
            postprocess: tensor outputs -> result
        """
    def __init__(self):
        self.input_size = 416
        self.default_output_path = "src/worker/inference/models/yolov4/output.jpg"

        self.anchors_path = "src/worker/inference/models/yolov4/yolov4_anchors.txt"
        self.class_names_path = "src/worker/inference/models/yolov4/coco.names"
        self.strides = np.array([8, 16, 32], dtype=np.int32)
        self.xyscale = [1.2, 1.1, 1.05]

        self.score_threshold = 0.25
        self.iou_threshold = 0.213
        self.nms_method = "nms"

        with open(self.anchors_path, 'r') as f:
            self.anchors = f.readline()
        self.anchors = np.array(self.anchors.split(','), dtype=np.float32)
        self.anchors = self.anchors.reshape(3, 3, 2)
        self.names: dict[int, str] = {}
        with open(self.class_names_path, 'r') as f:
            for ID, name in enumerate(f):
                self.names[ID] = name.strip('\n')

        pass

    def _image_preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        ih = iw = self.input_size
        h, w, _ = image_rgb.shape
        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image_rgb, (nw, nh))

        image_padded = np.full((ih, iw, 3), 128.0, dtype=np.float32)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2
        image_padded[dh:dh + nh, dw:dw + nw, :] = image_resized.astype(np.float32)
        image_padded = image_padded / 255.0
        return image_padded

    def preprocess(self, items: list[RawItem], meta: Optional[dict[str, Any]] = None) -> dict[str, np.ndarray]:
        imgs = []
        for i in items:
            bgr = None
            if i.type == "image_path":
                bgr = cv2.imread(i.data)
                if bgr is None:
                    raise ValueError(f"Failed to read image path: {i.data}")
            elif i.type == "image_bytes":
                bgr = np.frombuffer(i.data, dtype=np.uint8)
                bgr = cv2.imdecode(bgr, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError(f"Failed to decode image bytes: {i.data}")
            else:
                raise ValueError(f"Unsupported raw item type for yolov4: {i.type}")
            assert bgr is not None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            imgs.append(self._image_preprocess(rgb))
        batch = np.stack(imgs, axis=0).astype(np.float32)
        return {"input_1:0": batch}

    @staticmethod
    def _get_anchors(anchors_path: str) -> np.ndarray:
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = np.array(anchors.split(','), dtype=np.float32)
        return anchors.reshape(3, 3, 2)

    @staticmethod
    def _postprocess_bbbox(pred_bbox: list[np.ndarray], ANCHORS, STRIDES, XYSCALE):
        for i, pred in enumerate(pred_bbox):
            conv_shape = pred.shape
            output_size = conv_shape[1]
            conv_raw_dxdy = pred[:, :, :, :, 0:2]
            conv_raw_dwdh = pred[:, :, :, :, 2:4]
            xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
            xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)

            xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
            xy_grid = xy_grid.astype(np.float64)

            pred_xy = ((special.expit(conv_raw_dxdy) * XYSCALE[i]) - 0.5 * (XYSCALE[i] - 1) + xy_grid) * STRIDES[i]
            pred_wh = (np.exp(conv_raw_dwdh) * ANCHORS[i])
            pred[:, :, :, :, 0:4] = np.concatenate([pred_xy, pred_wh], axis=-1)

        pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = np.concatenate(pred_bbox, axis=0)
        return pred_bbox

    @staticmethod
    def _postprocess_boxes(pred_bbox, org_img_shape, input_size, score_threshold):
        valid_scale = [0, np.inf]
        pred_bbox = np.array(pred_bbox)

        pred_xywh = pred_bbox[:, 0:4]
        pred_conf = pred_bbox[:, 4]
        pred_prob = pred_bbox[:, 5:]

        # (x, y, w, h) -> (xmin, ymin, xmax, ymax)
        pred_coor = np.concatenate(
            [
                pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5,
            ],
            axis=-1,
        )

        org_h, org_w = org_img_shape
        resize_ratio = min(input_size / org_w, input_size / org_h)
        dw = (input_size - resize_ratio * org_w) / 2
        dh = (input_size - resize_ratio * org_h) / 2

        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        pred_coor = np.concatenate(
            [
                np.maximum(pred_coor[:, :2], [0, 0]),
                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1]),
            ],
            axis=-1,
        )

        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > score_threshold
        mask = np.logical_and(scale_mask, score_mask)

        coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]
        return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    @staticmethod
    def _bboxes_iou(boxes1, boxes2):
        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)
        return ious

    def _nms(self, bboxes, iou_threshold, sigma=0.3, method="nms"):
        classes_in_img = list(set(bboxes[:, 5]))
        best_bboxes = []

        for cls in classes_in_img:
            cls_mask = (bboxes[:, 5] == cls)
            cls_bboxes = bboxes[cls_mask]

            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[:max_ind], cls_bboxes[max_ind + 1:]])
                iou = self._bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                weight = np.ones((len(iou),), dtype=np.float32)

                if method == "nms":
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                elif method == "soft-nms":
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))
                else:
                    raise ValueError("method must be 'nms' or 'soft-nms'")

                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] > 0.0
                cls_bboxes = cls_bboxes[score_mask]

        return best_bboxes

    @staticmethod
    def _draw_bbox(image_rgb: np.ndarray, bboxes, classes: dict[int, str], show_label=True):
        num_classes = len(classes)
        image_h, image_w, _ = image_rgb.shape
        hsv_tuples = [(1.0 * x / num_classes, 1.0, 1.0) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

        random.seed(0)
        random.shuffle(colors)
        random.seed(None)

        for bbox in bboxes:
            coor = np.array(bbox[:4], dtype=np.int32)
            fontScale = 0.5
            score = float(bbox[4])
            class_ind = int(bbox[5])
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
            cv2.rectangle(image_rgb, c1, c2, bbox_color, bbox_thick)

            if show_label:
                bbox_mess = f"{classes[class_ind]}: {score:.2f}"
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                cv2.rectangle(
                    image_rgb,
                    c1,
                    (c1[0] + t_size[0], c1[1] - t_size[1] - 3),
                    bbox_color,
                    -1,
                )
                cv2.putText(
                    image_rgb,
                    bbox_mess,
                    (c1[0], c1[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale,
                    (0, 0, 0),
                    bbox_thick // 2,
                    lineType=cv2.LINE_AA,
                )

        return image_rgb

    # Draw boxes on detected objects & save to tests/worker/model/yolov4/index_output.jpg
    def postprocess(self, outputs: dict[str, np.ndarray], meta: Optional[dict[str, Any]] = None) -> Any:
        # Meta override
        meta = meta or {}
        output_dir = meta.get("output_dir", "tests/worker/model/yolov4")
        output_path_tpl = meta.get("output_path_template", os.path.join(output_dir, "output_{i}.jpg"))
        score_th = float(meta.get("score_threshold", self.score_threshold))
        iou_th = float(meta.get("iou_threshold", self.iou_threshold))
        nms_method = str(meta.get("nms_method", self.nms_method))
        save_images = bool(meta.get("save_images", False))

        # 1. outputs dict -> detections list (3 scales) in correct order
        output_names = ['Identity:0', 'Identity_1:0', 'Identity_2:0']
        detections_all = [outputs[name] for name in output_names]
        B = int(detections_all[0].shape[0])
        for d in detections_all[1:]:
            if int(d.shape[0]) != B:
                raise ValueError("YOLO outputs have inconsistent batch dimension.")

        # 2. Check raw items to draw boxes on original inputs image
        items = meta.get("items")
        if items is None:
            raise ValueError("Raw items are required for YOLOv4 model postprocessing! (only raw item mode supported)")
        if len(items) != B:
            raise ValueError(f"YOLO outputs have inconsistent batch dimension with raw items: inputs-{len(items)} vs output-{B}")

        # 3. Draw boxes for each image
        results = []
        os.makedirs(output_dir, exist_ok=True)
        for i in range(B):
            # 3.1 load original image for item i
            it = items[i]
            if it.type == "image_path":
                bgr = cv2.imread(it.data)
                if bgr is None:
                    raise ValueError(f"Failed to read image path: {it.data}")
            elif it.type == "image_bytes":
                buf = np.frombuffer(it.data, dtype=np.uint8)
                bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise ValueError("Failed to decode image bytes")
            else:
                raise ValueError(f"Unsupported raw item type for yolov4 postprocess: {it.type}")

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            original_shape = rgb.shape[:2]  # (h, w)

            # 3.2 slice each output layer to this image (keep batch dim = 1)
            detections_i = [d[i:i + 1, ...] for d in detections_all]

            # 3.3 decode + filter + nms (same logic as B=1)
            pred_bbox = self._postprocess_bbbox(detections_i, self.anchors, self.strides, self.xyscale)
            bboxes = self._postprocess_boxes(pred_bbox, original_shape, self.input_size, score_th)
            bboxes = self._nms(bboxes, iou_th, method=nms_method)

            out_path = None
            if save_images:
                drawn = self._draw_bbox(rgb.copy(), bboxes, self.names)
                out_path = output_path_tpl.format(i=i)
                Image.fromarray(drawn).save(out_path)

            results.append({
                "index": i,
                "num_boxes": len(bboxes),
                "bboxes": bboxes,  # list of [xmin, ymin, xmax, ymax, score, cls]
                "output_path": out_path,  # None if save_images=False
            })

        return results



    def generate_dummy_inputs(self, batch_size: int = 1, seed: int = 42) -> dict[str, np.ndarray]:
        raise NotImplementedError
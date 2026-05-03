#!/usr/bin/env python3
"""YOLO v3 object detection"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """Uses YOLO v3 to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize YOLO"""
        self.model = K.models.load_model(model_path)

        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet outputs

        Args:
            outputs: list of numpy arrays
            image_size: numpy array [image_height, image_width]

        Returns:
            boxes, box_confidences, box_class_probs
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        img_h = image_size[0]
        img_w = image_size[1]

        for i, output in enumerate(outputs):
            grid_h = output.shape[0]
            grid_w = output.shape[1]
            anchors = self.anchors[i]

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            cx = np.arange(grid_w)
            cy = np.arange(grid_h)

            cx = np.tile(cx, grid_h).reshape(grid_h, grid_w)
            cy = np.tile(cy.reshape(-1, 1), grid_w)

            bx = (self.sigmoid(tx) + cx[..., np.newaxis]) / grid_w
            by = (self.sigmoid(ty) + cy[..., np.newaxis]) / grid_h

            bw = (anchors[:, 0] * np.exp(tw)) / input_w
            bh = (anchors[:, 1] * np.exp(th)) / input_h

            x1 = (bx - bw / 2) * img_w
            y1 = (by - bh / 2) * img_h
            x2 = (bx + bw / 2) * img_w
            y2 = (by + bh / 2) * img_h

            box = np.stack((x1, y1, x2, y2), axis=-1)
            boxes.append(box)

            confidence = self.sigmoid(output[..., 4:5])
            box_confidences.append(confidence)

            class_probs = self.sigmoid(output[..., 5:])
            box_class_probs.append(class_probs)

        return boxes, box_confidences, box_class_probs

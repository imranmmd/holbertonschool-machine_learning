#!/usr/bin/env python3
"""YOLO v3 object detection"""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """YOLO class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize"""
        self.model = K.models.load_model(model_path)

        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Sigmoid"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process outputs

        Returns:
            boxes, box_confidences, box_class_probs
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        image_h = image_size[0]
        image_w = image_size[1]

        for i, output in enumerate(outputs):
            grid_h = output.shape[0]
            grid_w = output.shape[1]
            anchor_boxes = output.shape[2]

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x = np.zeros((grid_h, grid_w, anchor_boxes))
            c_y = np.zeros((grid_h, grid_w, anchor_boxes))

            for row in range(grid_h):
                for col in range(grid_w):
                    c_x[row, col, :] = col
                    c_y[row, col, :] = row

            b_x = (self.sigmoid(t_x) + c_x) / grid_w
            b_y = (self.sigmoid(t_y) + c_y) / grid_h

            p_w = self.anchors[i, :, 0]
            p_h = self.anchors[i, :, 1]

            b_w = (p_w * np.exp(t_w)) / input_w
            b_h = (p_h * np.exp(t_h)) / input_h

            x1 = (b_x - b_w / 2) * image_w
            y1 = (b_y - b_h / 2) * image_h
            x2 = (b_x + b_w / 2) * image_w
            y2 = (b_y + b_h / 2) * image_h

            box = np.zeros(output[..., :4].shape)
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2
            boxes.append(box)

            box_confidences.append(self.sigmoid(output[..., 4:5]))
            box_class_probs.append(self.sigmoid(output[..., 5:]))

        return boxes, box_confidences, box_class_probs

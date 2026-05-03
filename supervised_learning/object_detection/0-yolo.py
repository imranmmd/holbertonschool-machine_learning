#!/usr/bin/env python3
"""YOLO v3 object detection class initialization"""

import tensorflow.keras as K


class Yolo:
    """Uses YOLO v3 to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class constructor

        Args:
            model_path: path to Darknet Keras model
            classes_path: path to file containing class names
            class_t: box score threshold
            nms_t: IOU threshold for non-max suppression
            anchors: numpy.ndarray of anchor boxes
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f.readlines()]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

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
        """Process model outputs"""
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

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes"""
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]

            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(boxes[i][mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes)
        box_classes = np.concatenate(box_classes)
        box_scores = np.concatenate(box_scores)

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Apply non-max suppression

        Returns:
            box_predictions, predicted_box_classes, predicted_box_scores
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        classes = np.unique(box_classes)

        for cls in classes:
            idxs = np.where(box_classes == cls)[0]

            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            order = np.argsort(cls_scores)[::-1]

            while len(order) > 0:
                i = order[0]

                box_predictions.append(cls_boxes[i])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[i])

                if len(order) == 1:
                    break

                rest = order[1:]

                x1 = np.maximum(cls_boxes[i, 0], cls_boxes[rest, 0])
                y1 = np.maximum(cls_boxes[i, 1], cls_boxes[rest, 1])
                x2 = np.minimum(cls_boxes[i, 2], cls_boxes[rest, 2])
                y2 = np.minimum(cls_boxes[i, 3], cls_boxes[rest, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)
                inter = inter_w * inter_h

                area_i = (
                    (cls_boxes[i, 2] - cls_boxes[i, 0]) *
                    (cls_boxes[i, 3] - cls_boxes[i, 1])
                )

                area_rest = (
                    (cls_boxes[rest, 2] - cls_boxes[rest, 0]) *
                    (cls_boxes[rest, 3] - cls_boxes[rest, 1])
                )

                union = area_i + area_rest - inter
                iou = inter / union

                order = rest[iou < self.nms_t]

        return (
            np.array(box_predictions),
            np.array(predicted_box_classes),
            np.array(predicted_box_scores)
        )

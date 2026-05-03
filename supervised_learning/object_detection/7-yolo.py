#!/usr/bin/env python3
"""YOLO v3 object detection"""

import os
import cv2
import numpy as np
import tensorflow.keras as K


class Yolo:
    """YOLO class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """Initialize YOLO"""
        self.model = K.models.load_model(model_path)

        with open(classes_path, "r") as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """Sigmoid activation"""
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
            gh, gw, ab = output.shape[:3]

            tx = output[..., 0]
            ty = output[..., 1]
            tw = output[..., 2]
            th = output[..., 3]

            cx = np.zeros((gh, gw, ab))
            cy = np.zeros((gh, gw, ab))

            for row in range(gh):
                for col in range(gw):
                    cx[row, col, :] = col
                    cy[row, col, :] = row

            bx = (self.sigmoid(tx) + cx) / gw
            by = (self.sigmoid(ty) + cy) / gh

            pw = self.anchors[i, :, 0]
            ph = self.anchors[i, :, 1]

            bw = (pw * np.exp(tw)) / input_w
            bh = (ph * np.exp(th)) / input_h

            x1 = (bx - bw / 2) * image_w
            y1 = (by - bh / 2) * image_h
            x2 = (bx + bw / 2) * image_w
            y2 = (by + bh / 2) * image_h

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

        return (np.concatenate(filtered_boxes),
                np.concatenate(box_classes),
                np.concatenate(box_scores))

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """Apply non-max suppression"""
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for cls in np.unique(box_classes):
            idx = np.where(box_classes == cls)[0]

            cls_boxes = filtered_boxes[idx]
            cls_scores = box_scores[idx]

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

                area1 = ((cls_boxes[i, 2] - cls_boxes[i, 0]) *
                         (cls_boxes[i, 3] - cls_boxes[i, 1]))
                area2 = ((cls_boxes[rest, 2] - cls_boxes[rest, 0]) *
                         (cls_boxes[rest, 3] - cls_boxes[rest, 1]))

                union = area1 + area2 - inter
                iou = inter / union

                order = rest[iou < self.nms_t]

        return (np.array(box_predictions),
                np.array(predicted_box_classes),
                np.array(predicted_box_scores))

    @staticmethod
    def load_images(folder_path):
        """Load images from folder"""
        images = []
        image_paths = []

        for file_name in os.listdir(folder_path):
            path = os.path.join(folder_path, file_name)
            img = cv2.imread(path)

            if img is not None:
                images.append(img)
                image_paths.append(path)

        return images, image_paths

    def preprocess_images(self, images):
        """Resize and normalize images"""
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        pimages = []
        image_shapes = []

        for image in images:
            image_shapes.append(image.shape[:2])

            resized = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )

            pimages.append(resized / 255.0)

        return np.array(pimages), np.array(image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """Display image with predictions"""
        img = image.copy()

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)

            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            text = "{} {:.2f}".format(
                self.class_names[box_classes[i]],
                box_scores[i]
            )

            cv2.putText(
                img,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, img)
        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists("detections"):
                os.makedirs("detections")

            cv2.imwrite(os.path.join("detections", file_name), img)

        cv2.destroyAllWindows()

    def predict(self, folder_path):
        """
        Predict objects in all images inside folder

        Returns:
            predictions, image_paths
        """
        images, image_paths = self.load_images(folder_path)
        pimages, image_shapes = self.preprocess_images(images)

        outputs = self.model.predict(pimages)

        predictions = []

        for i in range(len(images)):
            image_outputs = [output[i] for output in outputs]

            boxes, box_confidences, box_class_probs = self.process_outputs(
                image_outputs,
                image_shapes[i]
            )

            boxes, box_classes, box_scores = self.filter_boxes(
                boxes,
                box_confidences,
                box_class_probs
            )

            boxes, box_classes, box_scores = self.non_max_suppression(
                boxes,
                box_classes,
                box_scores
            )

            predictions.append((boxes, box_classes, box_scores))

            self.show_boxes(
                images[i],
                boxes,
                box_classes,
                box_scores,
                os.path.basename(image_paths[i])
            )

        return predictions, image_paths

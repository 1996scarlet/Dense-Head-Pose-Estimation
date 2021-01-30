#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import cv2
import tensorflow as tf
from functools import partial


class BaseTFLiteFaceAlignment():
    def __init__(self, model_path, num_threads=1):
        # tflite model init
        self._interpreter = tf.lite.Interpreter(model_path=model_path,
                                                num_threads=num_threads)
        self._interpreter.allocate_tensors()

        # model details
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        # shape details
        self._input_shape = tuple(input_details[0]["shape"][-2:])
        self._edge_size = self._input_shape[-1]
        self._trans_distance = self._edge_size / 2.0

        # inference helper
        self._set_input_tensor = partial(self._interpreter.set_tensor,
                                         input_details[0]["index"])
        self._get_camera_matrix = partial(self._interpreter.get_tensor,
                                          output_details[0]["index"])
        self._get_landmarks = partial(self._interpreter.get_tensor,
                                      output_details[1]["index"])

    def _preprocessing(self, img, bbox, factor=2.7):
        """Pre-processing of the BGR image. Adopting warp affine for face corp.

        Arguments
        ----------
        img {numpy.array} : the raw BGR image.
        bbox {numpy.array} : bounding box with format: {x1, y1, x2, y2, score}.

        Keyword Arguments
        ----------
        factor : max edge scale factor for bounding box cropping.

        Returns
        ----------
        inp : input tensor with NHWC format.
        M : warp affine matrix.
        """

        maximum_edge = max(bbox[2:4] - bbox[:2]) * factor
        scale = self._edge_size * 2.0 / maximum_edge
        center = (bbox[2:4] + bbox[:2]) / 2.0
        cx, cy = self._trans_distance - scale * center

        M = np.array([[scale, 0, cx], [0, scale, cy]])

        cropped = cv2.warpAffine(img, M, self._input_shape, borderValue=0.0)
        rgb = cropped[:, :, ::-1].astype(np.float32)

        cv2.normalize(rgb, rgb, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX)

        inp = rgb.transpose(2, 0, 1)[None]

        return inp, M

    def _inference(self, input_tensor):
        self._set_input_tensor(input_tensor)
        self._interpreter.invoke()

    def _decode_landmarks(self, iM):
        raise NotImplementedError()

    def _postprocessing(self, M):
        iM = cv2.invertAffineTransform(M)

        R = self._get_camera_matrix()[0]

        landmarks = self._decode_landmarks(iM)

        return landmarks, R

    def get_landmarks(self, image, detected_faces=None):
        """Predict the landmarks for each face present in the image.

        This function predicts a set of 68 2D or 3D images, one for each image present.
        If detect_faces is None the method will also run a face detector.

        Arguments
        ----------
        image {numpy.array} : The input image.

        Keyword Arguments
        ----------
        detected_faces {list of numpy.array} : list of bounding boxes, one for each
        face found in the image (default: {None}, format: {x1, y1, x2, y2, score})
        """

        for box in detected_faces:
            if box[2] - box[0] < 100:
                continue
            inp, M = self._preprocessing(image, box)
            self._inference(inp)

            yield self._postprocessing(M)


class DenseFaceReconstruction(BaseTFLiteFaceAlignment):
    def _decode_landmarks(self, iM):
        points = self._get_landmarks()[0]

        points *= iM[0][0]
        points[:, :2] += iM[:, -1]

        return points


class DepthFacialLandmarks(BaseTFLiteFaceAlignment):
    def _decode_landmarks(self, iM):
        points = self._get_landmarks()[0]

        points *= iM[0, 0]
        points += iM[:, -1]

        return points

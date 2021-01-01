#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import cv2
import tensorflow as tf
from functools import partial
import time


class DenseFaceReconstruction():
    def __init__(self, model_path):
        # tflite interpreter and helper functions init
        self._tflite_init(model_path=model_path, num_threads=1)
        self._trans_distance = self._input_shape[-1] / 2.0

        # offset matrix append
        self._col = np.ones((1, 68))

    def _tflite_init(self, **kwargs):
        # tflite model init
        self._interpreter = tf.lite.Interpreter(**kwargs)
        self._interpreter.allocate_tensors()

        # model details
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        # shape details
        self._input_shape = tuple(input_details[0]["shape"][-2:])

        # inference helper
        self._set_input_tensor = partial(self._interpreter.set_tensor,
                                         input_details[0]["index"])
        self._get_params = partial(self._interpreter.get_tensor,
                                   output_details[0]["index"])
        self._get_uv_shape = partial(self._interpreter.get_tensor,
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
        scale = self._trans_distance * 4.0 / maximum_edge
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

        return self._get_params()[0], self._get_uv_shape()[0]

    def _postprocessing(self, out, M):
        iM = cv2.invertAffineTransform(M)

        R, pts3d = out
        pose = cv2.decomposeProjectionMatrix(R)[-1]

        pts3d[0] -= 1
        pts3d[1] -= self._input_shape[0]
        pts3d[1] *= -1
        pts3d[2] = 1

        return (iM @ pts3d).T, pose

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
            inp, M = self._preprocessing(image, box)
            out = self._inference(inp)

            yield self._postprocessing(out, M)


if __name__ == '__main__':

    from TFLiteFaceDetector import UltraLightFaceDetecion
    import sys

    fd = UltraLightFaceDetecion("weights/RFB-320.tflite", conf_threshold=0.88)
    fa = DenseFaceReconstruction("weights/matrix_and_landmarks.tflite")

    cap = cv2.VideoCapture(sys.argv[1])
    color = (125, 255, 125)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        start_time = time.perf_counter()

        boxes, scores = fd.inference(frame)

        for landmarks, pose in fa.get_landmarks(frame, boxes):
            print(pose.flatten())
            for p in np.round(landmarks).astype(np.int):
                cv2.circle(frame, tuple(p), 1, color, 1, cv2.LINE_AA)

        print(time.perf_counter() - start_time)

        cv2.imshow("result", frame)
        if cv2.waitKey(0) == ord('q'):
            break

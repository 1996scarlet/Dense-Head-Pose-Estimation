#!/usr/bin/python3
# -*- coding:utf-8 -*-

from BaseTFLiteFaceAlignment import BaseTFLiteFaceAlignment


class TFLite3DFacialLandmarks(BaseTFLiteFaceAlignment):
    def __init__(self, model_path, num_threads=1):
        super().__init__(model_path, num_threads)

    def _decode_landmarks(self, iM):
        pts3d = self._get_landmarks()[0]

        pts3d[0] -= 1
        pts3d[1] -= self._edge_size
        pts3d[1] *= -1
        pts3d[2] = 1

        return (iM @ pts3d).T


if __name__ == '__main__':

    from TFLiteFaceDetector import UltraLightFaceDetecion
    import sys
    import cv2
    import numpy as np
    import time

    fd = UltraLightFaceDetecion("weights/RFB-320.tflite", conf_threshold=0.88)
    fa = TFLite3DFacialLandmarks("weights/matrix_and_landmarks.tflite")

    cap = cv2.VideoCapture(sys.argv[1])
    color = (125, 255, 125)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        boxes, scores = fd.inference(frame)

        start_time = time.perf_counter()

        for landmarks, pose in fa.get_landmarks(frame, boxes):
            print(pose.flatten())
            for p in np.round(landmarks).astype(np.int):
                cv2.circle(frame, tuple(p), 1, color, 1, cv2.LINE_AA)

        print(time.perf_counter() - start_time)

        cv2.imshow("result", frame)
        if cv2.waitKey(0) == ord('q'):
            break

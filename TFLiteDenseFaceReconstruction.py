#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
from BaseTFLiteFaceAlignment import BaseTFLiteFaceAlignment


class TFLiteDenseFaceReconstruction(BaseTFLiteFaceAlignment):
    def __init__(self, model_path, num_threads=1):
        super().__init__(model_path, num_threads)

    def _decode_landmarks(self, iM):
        pts3d = self._get_landmarks()[0]

        pts3d[0] -= 1
        pts3d[1] -= self._edge_size
        pts3d[1] *= -1

        deepth = pts3d[2:].copy()

        pts3d[2] = 1
        pts3d = iM @ pts3d

        deepth -= 1
        deepth *= iM[0][0] * 2
        deepth -= np.min(deepth)
        return np.concatenate((pts3d, deepth), axis=0)


if __name__ == '__main__':
    import sys
    import cv2
    import time

    from Sim3DR import RenderPipeline
    from TFLiteFaceDetector import UltraLightFaceDetecion

    fd = UltraLightFaceDetecion("weights/RFB-320.tflite", conf_threshold=0.88)
    fa = TFLiteDenseFaceReconstruction("weights/dense_face.tflite")
    render = RenderPipeline("weights/triangles.npy")

    cap = cv2.VideoCapture(sys.argv[1])

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        boxes, scores = fd.inference(frame)

        for landmarks, pose in fa.get_landmarks(frame, boxes):
            landmarks = landmarks.astype(np.float32)
            start_time = time.perf_counter()
            render(landmarks.T.copy(order='C'), frame)
            print(time.perf_counter() - start_time)

        cv2.imshow("result", frame)
        if cv2.waitKey(0) == ord('q'):
            break

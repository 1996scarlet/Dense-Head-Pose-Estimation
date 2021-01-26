#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import sys
import cv2
import service

fd = service.UltraLightFaceDetecion("weights/RFB-320.tflite",
                                    conf_threshold=0.52)
fa = service.DenseFaceReconstruction("weights/dense_face.tflite")
mr = service.TrianglesMeshRender("asset/render.so", "asset/triangles.npy")

frame = cv2.imread(sys.argv[1])

boxes, scores = fd.inference(frame)

feed = frame.copy()

for landmarks, pose in fa.get_landmarks(feed, boxes):
    landmarks = landmarks.astype(np.float32)
    mr.render(landmarks.copy(), frame)

cv2.imshow("result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

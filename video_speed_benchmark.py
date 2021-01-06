#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import sys
import cv2
import service
import time


fd = service.UltraLightFaceDetecion("weights/RFB-320.tflite",
                                    conf_threshold=0.92)
fa = service.DenseFaceReconstruction("weights/dense_face.tflite")
mr = service.TrianglesMeshRender("asset/render.so", "asset/triangles.npy")

cap = cv2.VideoCapture(sys.argv[1])
cap.set(cv2.CAP_PROP_POS_FRAMES, 760)

counter = 0
rate = cap.get(5)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # face detection
    start_time = time.perf_counter()
    boxes, scores = fd.inference(frame)
    detect_cost = time.perf_counter() - start_time

    # raw copy for reconstruction
    feed = frame.copy()

    start_time = time.perf_counter()

    for landmarks, pose in fa.get_landmarks(feed, boxes):
        landmarks = landmarks.astype(np.float32)
        mr.render(landmarks.T.copy(), frame)

    recon_cost = time.perf_counter() - start_time

    if counter % rate == 0:
        counter = 0
        print(f"Detection Cost: {detect_cost * 1000:.2f}ms;    " +
              f"Reconstruction and Render Cost: {recon_cost * 1000:.2f}ms")

    counter += 1

    # frame = frame[:400, 200:1080, :]

    cv2.imshow("result", frame)
    # cv2.imwrite(f'./draft/gif/img{counter:0>3}.png', frame)
    if cv2.waitKey(1) == ord('q'):
        break

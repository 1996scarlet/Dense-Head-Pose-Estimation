#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
import cv2
import numpy as np
import service


def draw_poly(frame, landmarks, color=(128, 255, 255), thickness=1):
    cv2.polylines(frame, [
        landmarks[:17],
        landmarks[17:22],
        landmarks[22:27],
        landmarks[27:31],
        landmarks[31:36]
    ], False, color, thickness=thickness)
    cv2.polylines(frame, [
        landmarks[36:42],
        landmarks[42:48],
        landmarks[48:60],
        landmarks[60:]
    ], True, color, thickness=thickness)


def main(args, color=(224, 255, 255)):
    fd = service.UltraLightFaceDetecion("weights/RFB-320.tflite",
                                        conf_threshold=0.96)

    if args.mode in ["sparse", "pose"]:
        fa = service.DepthFacialLandmarks("weights/sparse_face.tflite")
    else:
        fa = service.DenseFaceReconstruction("weights/dense_face.tflite")
        if args.mode == "mesh":
            mr = service.TrianglesMeshRender("asset/render.so",
                                             "asset/triangles.npy")

    cap = cv2.VideoCapture(args.filepath)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # face detection
        boxes, scores = fd.inference(frame)

        # raw copy for reconstruction
        feed = frame.copy()

        for landmarks, pose in fa.get_landmarks(feed, boxes):

            if args.mode == "sparse":
                landmarks = np.round(landmarks).astype(np.int)
                for p in landmarks:
                    cv2.circle(frame, tuple(p), 2, color, 0, cv2.LINE_AA)
                draw_poly(frame, landmarks, color=color)

            elif args.mode == "dense":
                landmarks = np.round(landmarks).astype(np.int)
                landmarks = landmarks[:2].T
                for p in landmarks[::6]:
                    cv2.circle(frame, tuple(p), 1, color, 0, cv2.LINE_AA)

            elif args.mode == "mesh":
                landmarks = landmarks.astype(np.float32)
                mr.render(landmarks.T.copy(), frame)

        cv2.imshow("result", frame)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Video demo script.')
    parser.add_argument('-f', '--filepath', type=str, required=True)
    parser.add_argument('-m', '--mode', type=str, default='sparse',
                        choices=['sparse', 'dense', 'mesh', 'pose'])

    args = parser.parse_args()
    main(args)

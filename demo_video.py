#!/usr/bin/python3
# -*- coding:utf-8 -*-

import argparse
import service
import cv2


def main(args, color=(224, 255, 255)):
    fd = service.UltraLightFaceDetecion("weights/RFB-320.tflite",
                                        conf_threshold=0.98)

    if args.mode in ["sparse", "pose"]:
        fa = service.DepthFacialLandmarks("weights/sparse_face.tflite")
    else:
        fa = service.DenseFaceReconstruction("weights/dense_face.tflite")
        if args.mode == "mesh":
            color = service.TrianglesMeshRender("asset/render.so",
                                                "asset/triangles.npy")

    handler = getattr(service, args.mode)
    cap = cv2.VideoCapture(args.filepath)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # face detection
        boxes, scores = fd.inference(frame)

        # raw copy for reconstruction
        feed = frame.copy()

        for results in fa.get_landmarks(feed, boxes):
            handler(frame, results, color)

        # cv2.imwrite(f'draft/gif/fish/img{counter:0>4}.jpg', frame)

        cv2.imshow("demo", frame)
        if cv2.waitKey(0) == ord("q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video demo script.")
    parser.add_argument("-f", "--filepath", type=str, required=True)
    parser.add_argument("-m", "--mode", type=str, default="sparse",
                        choices=["sparse", "dense", "mesh", "pose"])

    args = parser.parse_args()
    main(args)

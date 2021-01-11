import cv2
import numpy as np

from .TFLiteFaceDetection import UltraLightFaceDetecion
from .TFLiteFaceAlignment import DenseFaceReconstruction, DepthFacialLandmarks
from .CtypesMeshRender import TrianglesMeshRender


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


def sparse(frame, results, color):
    landmarks = np.round(results[0]).astype(np.int)
    for p in landmarks:
        cv2.circle(frame, tuple(p), 2, color, 0, cv2.LINE_AA)
    draw_poly(frame, landmarks, color=color)


def dense(frame, results, color):
    landmarks = np.round(results[0]).astype(np.int)
    landmarks = landmarks[:2].T
    for p in landmarks[::6]:
        cv2.circle(frame, tuple(p), 1, color, 0, cv2.LINE_AA)


def mesh(frame, results, color):
    landmarks = results[0].astype(np.float32)
    color.render(landmarks.T.copy(), frame)


def pose(frame, results, color):
    landmarks, pose = results
    print(pose)

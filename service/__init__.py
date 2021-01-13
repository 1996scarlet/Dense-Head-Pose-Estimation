import cv2
import numpy as np

from .TFLiteFaceDetection import UltraLightFaceDetecion
from .TFLiteFaceAlignment import DenseFaceReconstruction, DepthFacialLandmarks
from .CtypesMeshRender import TrianglesMeshRender


def decode_params(params):
    R1 = params[0:1, :3]
    R2 = params[1:2, :3]
    t3d = params[:, -1:]

    # liner normalize
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    # rotate matrix
    R = np.concatenate((r1, r2, r3), axis=0)

    # projection matrix
    P = np.concatenate((R, t3d), axis=1)
    euler = cv2.decomposeProjectionMatrix(P)[-1]

    return R, euler


def build_camera_box(rear_size, factor=np.sqrt(2)):
    rear_depth = 0
    front_size = front_depth = int(factor * rear_size)

    point_3d = np.array([
        [-rear_size, -rear_size, rear_depth],
        [-rear_size, rear_size, rear_depth],
        [rear_size, rear_size, rear_depth],
        [rear_size, -rear_size, rear_depth],
        [-rear_size, -rear_size, rear_depth],
        [-front_size, -front_size, front_depth],
        [-front_size, front_size, front_depth],
        [front_size, front_size, front_depth],
        [front_size, -front_size, front_depth],
        [-front_size, -front_size, front_depth]
    ], dtype=np.float32)

    return point_3d


def draw_projection(frame, R, ver, color, thickness=2):
    radius = np.max(np.max(ver, 0) - np.min(ver, 0)) / 2
    offset = np.mean(ver[:27], 0)

    point_3d = build_camera_box(radius)

    R = R[:2]
    R[1] *= -1

    point_2d = point_3d @ R.T + offset
    points = point_2d.astype(np.int32)

    cv2.polylines(frame, [points], True, color, thickness, cv2.LINE_AA)
    cv2.line(frame, tuple(points[1]), tuple(points[6]),
             color, thickness, cv2.LINE_AA)
    cv2.line(frame, tuple(points[2]), tuple(points[7]),
             color, thickness, cv2.LINE_AA)
    cv2.line(frame, tuple(points[3]), tuple(points[8]),
             color, thickness, cv2.LINE_AA)


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
    landmarks, params = results

    R, euler = decode_params(params)

    draw_projection(frame, R, landmarks, color)

    print(euler.flatten())

import cv2
import numpy as np

from .TFLiteFaceDetection import UltraLightFaceDetecion
from .TFLiteFaceAlignment import DenseFaceReconstruction, DepthFacialLandmarks
from .CtypesMeshRender import TrianglesMeshRender


def decode_params(params):
    # rotate matrix
    R = params[:3, :3].copy()

    # liner normalize
    R[:2] /= np.linalg.norm(R[:2], axis=1, keepdims=True)
    R[2] = np.cross(R[0], R[1])

    # projection matrix
    P = params.copy()
    P[:3, :3] = R.T

    # decompose
    euler = cv2.decomposeProjectionMatrix(P)[-1]

    return R, euler


def build_projection_matrix(rear_size, factor=np.sqrt(2)):
    rear_depth = 0
    front_size = front_depth = int(factor * rear_size)

    projections = np.array([
        [-rear_size, -rear_size, rear_depth],
        [-rear_size, rear_size, rear_depth],
        [rear_size, rear_size, rear_depth],
        [rear_size, -rear_size, rear_depth],
        [-front_size, -front_size, front_depth],
        [-front_size, front_size, front_depth],
        [front_size, front_size, front_depth],
        [front_size, -front_size, front_depth],
    ], dtype=np.float32)

    return projections


def draw_projection(frame, R, landmarks, color, thickness=2):
    # build projection matrix
    radius = np.max(np.max(landmarks, 0) - np.min(landmarks, 0)) / 2
    projections = build_projection_matrix(radius)
    
    # refine rotate matrix
    rotate_matrix = R[:2]
    rotate_matrix[1] *= -1

    # 3D -> 2D
    center = np.mean(landmarks[:27], axis=0)
    points = projections @ rotate_matrix.T + center
    points = points.astype(np.int32)

    # draw poly
    cv2.polylines(frame, np.take(points, [
        [0, 1], [1, 2], [2, 3], [3, 0],
        [0, 4], [1, 5], [2, 6], [3, 7],
        [4, 5], [5, 6], [6, 7], [7, 4]
    ], axis=0), False, color, thickness, cv2.LINE_AA)


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

    # rotate matrix and euler angle
    R, euler = decode_params(params)
    print(f"Pitch: {euler[0]}; Yaw: {euler[1]}; Roll: {euler[2]};")

    draw_projection(frame, R, landmarks, color)

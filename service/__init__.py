import cv2
import numpy as np

from .TFLiteFaceDetection import UltraLightFaceDetecion
from .TFLiteFaceAlignment import DenseFaceReconstruction, DepthFacialLandmarks
from .CtypesMeshRender import TrianglesMeshRender


def build_camera_box(rear_size=9e4, factor=4/3):
    rear_depth = 0
    front_size = front_depth = int(factor * rear_size)

    point_3d = np.array([
        [-rear_size, -rear_size, rear_depth, 1],
        [-rear_size, rear_size, rear_depth, 1],
        [rear_size, rear_size, rear_depth, 1],
        [rear_size, -rear_size, rear_depth, 1],
        [-rear_size, -rear_size, rear_depth, 1],
        [-front_size, -front_size, front_depth, 1],
        [-front_size, front_size, front_depth, 1],
        [front_size, front_size, front_depth, 1],
        [front_size, -front_size, front_depth, 1],
        [-front_size, -front_size, front_depth, 1]
    ], dtype=np.float32)

    return point_3d


def plot_pose_box(img, P, ver, color, line_width=2):
    point_3d = build_camera_box()

    P[1] *= -1
    point_2d = point_3d @ P[:2].T

    point_2d = point_2d - np.mean(point_2d[:4], 0) + np.mean(ver[:27], 0)
    point_2d = point_2d.astype(np.int32)

    # Draw all the lines
    cv2.polylines(img, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(img, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)


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
    landmarks, camera_matrix = results

    # K, rot_mat, trans_vec, _, _, _, euler = cv2.decomposeProjectionMatrix(camera_matrix)
    # rot_vec = cv2.Rodrigues(rot_mat)

    plot_pose_box(frame, camera_matrix, landmarks, color)

    # print(K, camera_matrix)

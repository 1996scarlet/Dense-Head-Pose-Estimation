# coding: utf-8
import numpy as np
from . import Sim3DR_Cython


def get_normal(vertices, triangles):
    normal = np.zeros_like(vertices, dtype=np.float32)
    Sim3DR_Cython.get_normal(normal, vertices, triangles,
                             vertices.shape[0], triangles.shape[0])
    return normal


def rasterize(vertices, triangles, colors, bg=None,
              height=None, width=None, channel=None,
              reverse=False):
    if bg is not None:
        height, width, channel = bg.shape
    else:
        assert height is not None and width is not None and channel is not None
        bg = np.zeros((height, width, channel), dtype=np.uint8)

    buffer = np.zeros((height, width), dtype=np.float32) - 1e8

    if colors.dtype != np.float32:
        colors = colors.astype(np.float32)
    Sim3DR_Cython.rasterize(bg, vertices, triangles, colors, buffer, triangles.shape[0], height, width, channel,
                            reverse=reverse)
    return bg


def _norm(arr): return arr / np.sqrt(np.sum(arr ** 2, axis=1))[:, None]


def norm_vertices(vertices):
    vertices -= vertices.min(0)[None, :]
    vertices /= vertices.max()
    vertices *= 2
    vertices -= vertices.max(0)[None, :] / 2
    return vertices


def convert_type(obj):
    if isinstance(obj, tuple) or isinstance(obj, list):
        return np.array(obj, dtype=np.float32)[None, :]
    return obj


class RenderPipeline(object):
    def __init__(self, file_path):
        self.color_ambient = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        self.color_directional = np.array([0.6, 0.6, 0.6], dtype=np.float32)

        self._light_pos = np.array([0, 0, 5], dtype=np.float32)
        self._triangles = np.load(file_path)

    def __call__(self, vertices, bg, texture=None):
        normal = get_normal(vertices, self._triangles)

        light = np.zeros_like(vertices, dtype=np.float32)
        light += self.color_ambient

        vertices_n = norm_vertices(vertices.copy())

        direction = _norm(self._light_pos - vertices_n)
        cos = np.sum(normal * direction, axis=1)[:, None]
        cos = np.clip(cos, 0, 1)

        light += self.color_directional * cos

        light = np.clip(light, 0, 1)

        if texture is None:
            texture = light
        else:
            texture *= light

        render_img = rasterize(vertices, self._triangles, texture, bg=bg)
        return render_img

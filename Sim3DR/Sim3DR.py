# coding: utf-8
import numpy as np
from . import Sim3DR_Cython


def _norm(arr): return arr / np.sqrt(np.sum(arr ** 2, axis=1))[:, None]


def norm_vertices(vertices):
    vertices -= vertices.min(0)[None, :]
    vertices /= vertices.max()
    vertices *= 2
    vertices -= vertices.max(0)[None, :] / 2
    return vertices


class RenderPipeline(object):
    def __init__(self, file_path):
        self._ambient = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        self._directional = np.array([0.6, 0.6, 0.6], dtype=np.float32)
        self._light_pos = np.array([0, 0, 5], dtype=np.float32)

        self._triangles = np.load(file_path)
        self._tri_nums = self._triangles.shape[0]

    def _get_normal(self, vertices):
        normal = np.zeros_like(vertices, dtype=np.float32)
        Sim3DR_Cython.get_normal(normal, vertices, self._triangles,
                                 vertices.shape[0], self._tri_nums)
        return normal

    def _rasterize(self, bg, vertices, colors, alpha=1):
        Sim3DR_Cython.rasterize(bg, vertices, self._triangles, colors,
                                self._tri_nums, *bg.shape, alpha)
        return bg

    def __call__(self, vertices, bg):
        normal = self._get_normal(vertices)

        light = np.zeros_like(vertices, dtype=np.float32)
        light += self._ambient

        vertices_n = norm_vertices(vertices.copy())

        direction = _norm(self._light_pos - vertices_n)
        cos = np.sum(normal * direction, axis=1)[:, None]
        cos = np.clip(cos, 0, 1)

        light += self._directional * cos

        light = np.clip(light, 0, 1).astype(np.float32)

        render_img = self._rasterize(bg, vertices, light)
        return render_img

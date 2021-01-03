# coding: utf-8
import numpy as np
from . import Sim3DR_Cython


class RenderPipeline(object):
    def __init__(self, file_path):
        self._ambient = np.array([0.3, 0.3, 0.3], dtype=np.float32)
        self._directional = np.array([0.6, 0.6, 0.6], dtype=np.float32).clip(0, 1)

        self._light_pos = np.array([1, 1, 5], dtype=np.float32)

        self._triangles = np.load(file_path)
        self._tri_nums = self._triangles.shape[0]

    def _get_normal(self, vertices):
        Sim3DR_Cython.get_normal(vertices, self._triangles,
                                 vertices.shape[0], self._tri_nums)

    def _rasterize(self, bg, vertices, colors, alpha=1):
        Sim3DR_Cython.rasterize(bg, vertices, self._triangles, colors,
                                self._tri_nums, *bg.shape, alpha)
        return bg

    def __call__(self, vertices, bg):
        vertices_n = vertices.copy()
        self._get_normal(vertices_n)

        light = np.zeros_like(vertices, dtype=np.float32)
        light += self._ambient

        cos = np.sum(vertices_n, axis=1).copy()
        cos = np.clip(cos[:, None], 0, 1)

        light += self._directional * cos
        light = light.astype(np.float32)

        render_img = self._rasterize(bg, vertices, light)
        return render_img

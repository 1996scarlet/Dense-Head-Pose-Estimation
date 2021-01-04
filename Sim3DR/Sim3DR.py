# coding: utf-8
import numpy as np
from . import Sim3DR_Cython


class RenderPipeline(object):
    def __init__(self, file_path):
        self._light = np.array([1, 1, 5], dtype=np.float32)
        self._directional = np.array([0.6, 0.6, 0.6], dtype=np.float32)
        self._ambient = np.array([0.3, 0.3, 0.3], dtype=np.float32)

        self._triangles = np.load(file_path)
        self._tri_nums = self._triangles.shape[0]

    def __call__(self, vertices, bg):
        Sim3DR_Cython.render(vertices, vertices.shape[0],
                             self._triangles, self._tri_nums,
                             self._light,
                             self._directional,
                             self._ambient,
                             bg, *bg.shape)

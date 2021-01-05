# coding: utf-8
import numpy as np
import ctypes


class RenderPipeline(object):
    def __init__(self, file_path):
        self._light = np.array([1, 1, 5], dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._directional = np.array([0.6, 0.6, 0.6], dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._ambient = np.array([0.3, 0.3, 0.3], dtype=np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        self._triangles = np.load(file_path)
        self._tri_nums = self._triangles.shape[0]
        
        self._triangles = self._triangles.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self._render = ctypes.CDLL("Sim3DR/render.so")

    def __call__(self, vertices, bg):
        self._render._render(vertices.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                             vertices.shape[0],
                             self._triangles,
                             self._tri_nums,
                             self._light,
                             self._directional,
                             self._ambient,
                             bg.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
                             *bg.shape)

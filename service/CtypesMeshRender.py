#!/usr/bin/python3
# -*- coding:utf-8 -*-

import numpy as np
import ctypes


class TrianglesMeshRender():

    def __init__(self,
                 clibs,
                 triangles,
                 light=[1, 1, 5],
                 direction=[0.6, 0.6, 0.6],
                 ambient=[0.6, 0.5, 0.4]):

        self._clibs = ctypes.CDLL(clibs)

        self._triangles = np.load(triangles)
        self._tri_nums = self._triangles.shape[0]
        self._triangles = np.ctypeslib.as_ctypes(self._triangles)

        self._light = np.array(light, dtype=np.float32)
        self._light = np.ctypeslib.as_ctypes(self._light)

        self._direction = np.array(direction, dtype=np.float32)
        self._direction = np.ctypeslib.as_ctypes(self._direction)

        self._ambient = np.array(ambient, dtype=np.float32)
        self._ambient = np.ctypeslib.as_ctypes(self._ambient)

    def render(self, vertices, bg):
        self._clibs._render(
            self._triangles, self._tri_nums,
            self._light, self._direction, self._ambient,
            np.ctypeslib.as_ctypes(vertices),
            vertices.shape[0],
            np.ctypeslib.as_ctypes(bg),
            bg.shape[0], bg.shape[1]
        )

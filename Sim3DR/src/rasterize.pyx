#cython: language_level=3
cimport numpy as np
cimport cython

# use the Numpy-C-API from Cython
np.import_array()

# cdefine the signature of our c function
cdef extern from "render.h":
    void _render(float *vertices, int nver, int *triangles, int ntri,
                     float *light, float *directional, float *ambient,
                     unsigned char*image, int h, int w, int c, float alpha)


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def render(np.ndarray[float, ndim=2, mode = "c"] vertices not None,
                   int nver, 
                   np.ndarray[int, ndim=2, mode="c"] triangles not None,
                   int ntri,
                   np.ndarray[float, ndim=1, mode = "c"] light not None,
                   np.ndarray[float, ndim=1, mode = "c"] directional not None,
                   np.ndarray[float, ndim=1, mode = "c"] ambient not None,
                   np.ndarray[unsigned char, ndim=3, mode = "c"] image not None,
                   int h, int w, int c, float alpha = 1):
    _render(
        <float*> np.PyArray_DATA(vertices), nver,
        <int*> np.PyArray_DATA(triangles), ntri,
        <float*> np.PyArray_DATA(light),
        <float*> np.PyArray_DATA(directional),
        <float*> np.PyArray_DATA(ambient),
        <unsigned char*> np.PyArray_DATA(image),
        h, w, c, alpha)

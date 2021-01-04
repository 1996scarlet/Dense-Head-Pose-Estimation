#cython: language_level=3
cimport numpy as np
cimport cython

# use the Numpy-C-API from Cython
np.import_array()

# cdefine the signature of our c function
cdef extern from "render.h":
    void _render(const float *vertices, int nver,
                 const int *triangles, int ntri,
                 const float *light,
                 const float *directional,
                 const float *ambient,
                 unsigned char*image, int h, int w, int c)


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
                   int h, int w, int c):
    _render(
        <const float*> np.PyArray_DATA(vertices), nver,
        <const int*> np.PyArray_DATA(triangles), ntri,
        <const float*> np.PyArray_DATA(light),
        <const float*> np.PyArray_DATA(directional),
        <const float*> np.PyArray_DATA(ambient),
        <unsigned char*> np.PyArray_DATA(image),
        h, w, c)

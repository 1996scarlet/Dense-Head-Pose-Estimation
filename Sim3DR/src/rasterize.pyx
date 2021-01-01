import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool

# use the Numpy-C-API from Cython
np.import_array()

# cdefine the signature of our c function
cdef extern from "rasterize.h":
    void _rasterize_triangles(
            float*vertices, int*triangles, float*depth_buffer, int*triangle_buffer, float*barycentric_weight,
            int ntri, int h, int w
    )

    void _rasterize(
            unsigned char*image, float*vertices, int*triangles, float*colors, float*depth_buffer,
            int ntri, int h, int w, int c, float alpha, bool reverse
    )

    void _get_tri_normal(float *tri_normal, float *vertices, int *triangles, int nver, bool norm_flg)
    void _get_ver_normal(float *ver_normal, float*tri_normal, int*triangles, int nver, int ntri)
    void _get_normal(float *ver_normal, float *vertices, int *triangles, int nver, int ntri)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_tri_normal(np.ndarray[float, ndim=2, mode="c"] tri_normal not None,
                   np.ndarray[float, ndim=2, mode = "c"] vertices not None,
                   np.ndarray[int, ndim=2, mode="c"] triangles not None,
                   int ntri, bool norm_flg = False):
    _get_tri_normal(<float*> np.PyArray_DATA(tri_normal), <float*> np.PyArray_DATA(vertices),
                    <int*> np.PyArray_DATA(triangles), ntri, norm_flg)

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def get_ver_normal(np.ndarray[float, ndim=2, mode = "c"] ver_normal not None,
                   np.ndarray[float, ndim=2, mode = "c"] tri_normal not None,
                   np.ndarray[int, ndim=2, mode="c"] triangles not None,
                   int nver, int ntri):
    _get_ver_normal(
        <float*> np.PyArray_DATA(ver_normal), <float*> np.PyArray_DATA(tri_normal), <int*> np.PyArray_DATA(triangles),
        nver, ntri)

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def get_normal(np.ndarray[float, ndim=2, mode = "c"] ver_normal not None,
                   np.ndarray[float, ndim=2, mode = "c"] vertices not None,
                   np.ndarray[int, ndim=2, mode="c"] triangles not None,
                   int nver, int ntri):
    _get_normal(
        <float*> np.PyArray_DATA(ver_normal), <float*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles),
        nver, ntri)


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def rasterize_triangles(
        np.ndarray[float, ndim=2, mode = "c"] vertices not None,
        np.ndarray[int, ndim=2, mode="c"] triangles not None,
        np.ndarray[float, ndim=2, mode = "c"] depth_buffer not None,
        np.ndarray[int, ndim=2, mode = "c"] triangle_buffer not None,
        np.ndarray[float, ndim=2, mode = "c"] barycentric_weight not None,
        int ntri, int h, int w
):
    _rasterize_triangles(
        <float*> np.PyArray_DATA(vertices), <int*> np.PyArray_DATA(triangles),
        <float*> np.PyArray_DATA(depth_buffer), <int*> np.PyArray_DATA(triangle_buffer),
        <float*> np.PyArray_DATA(barycentric_weight),
        ntri, h, w)

@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def rasterize(np.ndarray[unsigned char, ndim=3, mode = "c"] image not None,
              np.ndarray[float, ndim=2, mode = "c"] vertices not None,
              np.ndarray[int, ndim=2, mode="c"] triangles not None,
              np.ndarray[float, ndim=2, mode = "c"] colors not None,
              np.ndarray[float, ndim=2, mode = "c"] depth_buffer not None,
              int ntri, int h, int w, int c, float alpha = 1, bool reverse = False
              ):
    _rasterize(
        <unsigned char*> np.PyArray_DATA(image), <float*> np.PyArray_DATA(vertices),
        <int*> np.PyArray_DATA(triangles),
        <float*> np.PyArray_DATA(colors),
        <float*> np.PyArray_DATA(depth_buffer),
        ntri, h, w, c, alpha, reverse)

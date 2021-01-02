'''
To compile:
python3 setup.py build_ext --inplace
'''

from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy

setup(
    name='Sim3DR_Cython',  # not the package name
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("Sim3DR_Cython",
                           sources=["src/rasterize.pyx",
                                    "src/rasterize_kernel.cpp"],
                           language='c++',
                           include_dirs=[numpy.get_include()],
                           extra_compile_args=["-std=c++11"])],
)

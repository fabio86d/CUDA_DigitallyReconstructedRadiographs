from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("SiddonGpuPy",
                             sources=["SiddonGpuPy.pyx"],
                             include_dirs=[numpy.get_include(), 
                                           "C:\\Users\\fabio\\Documents\\Programming\\Python\\Siddon\\noCmake\\VoidReturn\\ReleaseVersion08_08_2017\\SiddonPythonModule\\include"
                                           "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\include"],
                             library_dirs = ["C:\\Users\\fabio\\Documents\\Programming\\Python\\Siddon\\noCmake\\VoidReturn\\ReleaseVersion08_08_2017\\SiddonPythonModule\\lib",
                                             "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v8.0\\lib\\x64"],
                             libraries = ["SiddonGpu", "cudart_static"],
                             language = "c++")]
)
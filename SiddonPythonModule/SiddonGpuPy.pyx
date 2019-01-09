import cython

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

# declare the interface to the C++ code
cdef extern from "include\siddon_class.cuh" :
    cdef cppclass SiddonGpu :
        SiddonGpu()
        SiddonGpu(int *NumThreadsPerBlock,
                  float *movImgArray,
                  int *MovSize,
                  float *MovSpacing,
                  float X0, float Y0, float Z0,
                  int *DRRSize)
        void generateDRR(float *source,
                         float *DestArray,
                         float *drrArray)

cdef class pySiddonGpu :
    cdef SiddonGpu* thisptr # hold a C++ instance
    cdef DRRsize
    def __cinit__(self, np.ndarray[int, ndim = 1, mode = "c"] NumThreadsPerBlock not None,
                        np.ndarray[float, ndim = 1, mode = "c"] movImgArray not None,
                        np.ndarray[int, ndim = 1, mode = "c"] MovSize not None,
                        np.ndarray[float, ndim = 1, mode = "c"] MovSpacing not None,
                        X0, Y0, Z0,
                        np.ndarray[int, ndim = 1, mode = "c"] DRRsize not None) :

        self.DRRsize = DRRsize
        self.thisptr = new SiddonGpu(&NumThreadsPerBlock[0],
                                     &movImgArray[0],
                                     &MovSize[0],
                                     &MovSpacing[0],
                                     X0, Y0, Z0,
                                     &DRRsize[0])

    def generateDRR(self, np.ndarray[float, ndim = 1, mode = "c"] source not None,
                          np.ndarray[float, ndim = 1, mode = "c"] DestArray not None) :

        # generate contiguous output array
        drrArray = np.zeros(self.DRRsize[0] * self.DRRsize[1] * self.DRRsize[2], dtype = np.float32, order = 'C')
        cdef float[::1] cdrrArray = drrArray

        self.thisptr.generateDRR(&source[0], &DestArray[0], &cdrrArray[0])

        return cdrrArray

    def delete(self) :
        if self.thisptr is not NULL :
            "C++ object being destroyed"
            del self.thisptr


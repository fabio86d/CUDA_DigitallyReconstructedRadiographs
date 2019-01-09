# GPU accelerated generation of digitally reconstructed radiographs (DRR)

# Brief Description
The repository generates a Python library (SiddonGpuPy.pyd) for generating DRRs from a CT/MRI scan with parallelization on the GPU using CUDA (Invidia). 
The algorithm is based on the method proposed by Siddon ("Fast calculation of the exact radiological path for a three-dimensional CT array.", 1985 Med. Phys.) and 
on the improvements proposed by Greef et al. ("Accelerated ray tracing for radiotherapy dose calculations on a GPU.", 2009 Med. Phys.).
The original library written in C-C++ and parallelized using the CUDA toolkit (https://developer.nvidia.com/cuda-zone) is wrapped to a Python library using Cython.

A description of the parallelized algorithm for DRR generation can be found in the 5th Chapter of my PhD thesis (Chapter5_HipHop_2D3Dregistration.pdf).

The Python library SiddonGpuPy.pyd is used in the HipHop_2D3Dregistration framework (see repository https://github.com/fabio86d/HipHop_2D3Dregistration.git) 
for 2D/3D registration between volumetric medical images (CT/MRI scan) and X-ray images.

This repository also shows the process of generating a C++ library and wrapping it to a Python library using Cython.

# Procedure to generate Cpp library

1) Use Cmake to generate a Visual Studio project that builds the Cpp static library (SiddonGpu.lib) 

The source folder for Cmake is SiddonClassLib\src, and includes a Cmake file (CMakeLists.txt), a header cuda file (.cuh) and a source cuda file (.cu) of the siddon_class.
The folder where to build the binaries is SiddonClassLib\bin.

2) Open the created Visual Studio project and BUILD ALL in Release mode. The SiddonGpu.lib file will be generated in the Release folder.

# Procedure to wrap a Cpp library to a Python library using Cython.

The SiddonPythonModule folder includes:
- include/siddon_class.cuh (header cuda file with which SiddonGpu.lib was built)
- lib/SiddonGpu.lib (library generated with Cmake and Visual Studio, see above)
- a setup.py file (the python script that needs to be run)
- a  SiddonGpuPy.pyx (it represents the interface between the Cpp class and the Python class)

In order to wrap the SiddonGpu.lib library to a Python library (SiddonGpuPy.pyd):
from the SiddonPythonModule directory, just run
	"python setup.py build_ext --inplace"

# Use of the package

In the HipHop_2D3Dregistration framework (https://github.com/fabio86d/HipHop_2D3Dregistration.git) the SiddonGpuPy is imported in the ProjectorsModule 
and used to fast generate DRR from a CT scan of the pelvis.
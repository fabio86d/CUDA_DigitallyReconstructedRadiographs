Release Version 08.08.2017

Python extension module for Siddon Class.

- Initialization of the class loads the CT scan onto the GPU device.
- the function member generateDRR produces a DRR
- IMPORTANT: the Delete function must be manually called to delete the c++ object.

To build the Python extension:

put the Release version of the library into \lib subfolder
put the .cuh header with which the library was built into \include subfolder
make sure the paths are correct in SiddonGpuPy.pyx file

cmd
cd current directory
python setup.py build_ext --inplace
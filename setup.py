from setuptools import setup, Extension
import numpy as np

opencl_ops_module = Extension('opencl_ops',
                            sources=['opencl_ops.cpp'],
                            include_dirs=[np.get_include(), '/usr/local/include'],
                            library_dirs=['/usr/local/lib'],
                            libraries=['OpenCL'],
                            extra_compile_args=['-std=c++11'])

setup(name='opencl_ops',
      version='1.0',
      description='Python bindings for OpenCL Operations',
      ext_modules=[opencl_ops_module])
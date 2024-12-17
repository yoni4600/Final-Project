# coding:utf-8

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy
import sys
import platform

# Adjust compile and link flags based on the platform
extra_compile_args = []
extra_link_args = []

if platform.system() == "Windows":
    # Use MSVC compatible flags
    extra_compile_args.extend(['/std:c++14', '/Zi', '/Od'])  # Use C++14 and add debug flags
    extra_link_args.extend(['/DEBUG'])                      # Enable debugging symbols
else:
    # Use GCC/Clang compatible flags
    extra_compile_args.extend(['-std=c++14', '-g', '-O0'])  # Enable C++14 with debugging
    extra_link_args.extend(['-g'])

# Build setup
setup(
    ext_modules=cythonize(Extension(
        'necython',
        sources=[
            'necython/extension.pyx',
            'necython/cpp/aco.cpp',
            'necython/cpp/common.cpp',
            'necython/cpp/sampling.cpp',
            'necython/cpp/walker.cpp',
        ],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
        language='c++',
        include_dirs=[numpy.get_include(), "necython/cpp"],  # Add include directories
    ), compiler_directives={'language_level': "3"}),
)

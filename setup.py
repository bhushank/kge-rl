from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("/home/mitarb/kotnis/Code//kge-rl/sample_list.pyx")
)

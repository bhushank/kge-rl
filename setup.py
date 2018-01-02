from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("/home/kotnis/code/kge-rl/sample_list.pyx")
)

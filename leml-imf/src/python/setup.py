from distutils.core import setup, Extension
import numpy

ALLLIB = ["mf.o", "multilabel.o", "bilinear.o", "multiple_linear.o", "smat.o",
            "dmat.o", "tron.o", "zlib_util.o", "zlib/libz.a"]

#            "dmat.o", "tron.o", "blas/blas.a", "zlib_util.o", "zlib/libz.a"]

CFLAGS = ["-Wall", "-Wconversion", "-O3", "-fPIC", "-fopenmp", "-std=gnu++0x"]

objects = ["../{0}".format(l) for l in ALLLIB]
#objects = ["memcpy.o"] + objects
ELA = []
module1 = Extension("__train_mf", ["train_mf.cpp"], 
                extra_objects=objects,
                extra_compile_args=CFLAGS,
                runtime_library_dirs=["/root/rs/libs"],
                extra_link_args=ELA,
                library_dirs=["/root/rs/libs"],
                libraries=["blas", "lapack", "gfortran", "gomp"]
                )
#libraries=["lapack_atlas", "f77blas", "cblas", "atlas", "gfortran", "gomp"]

setup (name="__train_mf", version = "1.0", 
        description = "train inductive matix factorization", 
        include_dirs=[numpy.get_include()],
        ext_modules = [module1])




# Intel MKL
ifdef MKLROOT
CXX = icpc
CFLAGS = -Wall -Wconversion -O3 -fPIC -openmp -std=c++11 -I${MKLROOT}/include
BLASFLAGS = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm
else
CXX = g++
#CFLAGS = -Wall -Wconversion -g -fPIC -fopenmp -std=c++11
CFLAGS = -Wall -Wconversion -O3 -fPIC -fopenmp -std=c++11
#BLASFLAGS = -llapack_atlas -lf77blas -lcblas -latlas -lgfortran -L/u/rofuyu/.local/lib
BLASFLAGS = -lblas -llapack
endif


# ======================== flags for ICC on stampede ===========================
#CXX = icc
#CC = icc
#ICC_CFLAGS = -Wall -Wconversion -O3 -fPIC -openmp -std=c++11
#ICC_BLAS = -I$MKLROOT/include -mkl=parallel -openmp 
#BLASFLAGS = $(ICC_BLAS)
#CFLAGS = $(ICC_CFLAGS)

# ======================== flags for G++ on stampede ===========================
# module swap intel gcc; module load mkl 
#CXX = g++ 
#CC = gcc
#GCC_CFLAGS = -Wall -Wconversion -O3 -fPIC -fopenmp -std=c++11
#GCC_BLAS = -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
#BLASFLAGS = ${GCC_BLAS}
#CFLAGS = ${GCC_CFLAGS}

LIBSFLAGS= $(BLASFLAGS) #-lz -L../src/zlib

ALLLIB = ../src/dmat.o ../src/smat.o ../src/mf.o ../src/bilinear.o ../src/multiple_linear.o ../src/tron.o ../src/zlib_util.o ../src/zlib/libz.a
INC = -I../src

all: test-imf

test-imf: test-imf.cpp
	$(CXX) $(CFLAGS) $(INC) -o test-imf test-imf.cpp $(ALLLIB) $(LIBSFLAGS)

new: clean all

clean:
	rm -f test-imf

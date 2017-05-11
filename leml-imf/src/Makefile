

ifdef MKLROOT
CXX = icpc
CC = icc
CFLAGS = -Wall -Wconversion -O3 -fPIC -openmp -std=c++11 -I${MKLROOT}/include
BLASFLAGS = -Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_intel_thread.a -Wl,--end-group -lpthread -lm
else
CXX ?= g++
CC ?= gcc
#CFLAGS = -Wall -Wconversion -g -fPIC -fopenmp -std=gnu++0x
CFLAGS = -Wall -Wconversion -O3 -fPIC -fopenmp -std=gnu++0x
#BLASFLAGS = -llapack_atlas -lf77blas -lcblas -latlas -lgfortran -L/u/rofuyu/.local/lib
BLASFLAGS = -lblas -llapack
endif

# ======================== flags for ICC on stampede ===========================
#CXX = icc
#CC = icc
#ICC_CFLAGS = -Wall -Wconversion -O3 -fPIC -openmp -std=gnu++0x
#ICC_BLAS = -I$MKLROOT/include -mkl=parallel -openmp 
#BLASFLAGS = $(ICC_BLAS)
#CFLAGS = $(ICC_CFLAGS)

# ======================== flags for G++ on stampede ===========================
# module swap intel gcc; module load mkl 
#CXX = g++ 
#CC = gcc
#GCC_CFLAGS = -Wall -Wconversion -O3 -fPIC -fopenmp -std=gnu++0x
#GCC_BLAS = -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm
#BLASFLAGS = ${GCC_BLAS}
#CFLAGS = ${GCC_CFLAGS}

LIBSFLAGS= $(BLASFLAGS) # -lz -Lzlib 
OS = $(shell uname)

ALLLIB = mf.o multilabel.o bilinear.o multiple_linear.o smat.o dmat.o tron.o zlib_util.o zlib/libz.a

new: clean lib

all: train-ml 
lib: $(ALLLIB)

train-ml: train-ml.cpp multilabel.o bilinear.o multiple_linear.o smat.o dmat.o zlib_util.o tron.o $(ALLLIB)
	$(CXX) $(CFLAGS) -o train-ml train-ml.cpp $(ALLLIB) $(LIBSFLAGS) 

multilabel.o: multilabel.cpp multilabel.h bilinear.h multiple_linear.h tron.h smat.h dmat.h
	$(CXX) $(CFLAGS) -c -o multilabel.o multilabel.cpp
#$(LIBSFLAGS)

mf.o: mf.cpp mf.h bilinear.h multiple_linear.h tron.h smat.h dmat.h
	$(CXX) $(CFLAGS) -c -o mf.o mf.cpp
#$(LIBSFLAGS)

bilinear.o: bilinear.cpp bilinear.h tron.h smat.h
	$(CXX) $(CFLAGS) -c -o bilinear.o bilinear.cpp
#$(LIBSFLAGS)

multiple_linear.o: multiple_linear.h multiple_linear.cpp
	${CXX} ${CFLAGS} -c -o multiple_linear.o multiple_linear.cpp

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

smat.o: smat.h smat.cpp 
	${CXX} ${CFLAGS} -c -o smat.o smat.cpp

dmat.o: dmat.h dmat.cpp 
	${CXX} ${CFLAGS} -c -o dmat.o dmat.cpp

zlib_util.o: zlib_util.h zlib_util.cpp
	${CXX} ${CFLAGS} -c -o zlib_util.o zlib_util.cpp

zlib/libz.a: 
	make -C zlib

clean:
	rm -f *~ *.o train-ml 
	make -C zlib clean
	make -C matlab clean

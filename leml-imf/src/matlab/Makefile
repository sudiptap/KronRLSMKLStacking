
# TACC
ifdef MKLROOT
MATLAB = /work/apps/matlab/2015a
CC = icpc
INC = -I$(MATLAB)/extern/include -I$(MATLAB)/simulink/include -I$(MKLROOT)/include

MKLLIB += $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_intel_thread.a $(MKLROOT)/lib/intel64/libmkl_core.a

OPTS += -D_GNU_SOURCE -DMATLAB_MEX_FILE -ansi -fexceptions -fPIC -fno-omit-frame-pointer -pthread -openmp -O3 -DNDEBUG -std=c++11 -xHost
#-ipo

#LIBS = -shared -Wl,--version-script,$(MATLAB)/extern/lib/glnxa64/mexFunction.map -Wl,--no-undefined 
LIBS = -pthread -Wl,--no-undefined -openmp -Wall -shared -O3 -Wl,--version-script,$(MATLAB)/extern/lib/glnxa64/mexFunction.map
LIBS += -Wl,-rpath-link,$(MATLAB)/bin/glnxa64 -L$(MATLAB)/bin/glnxa64 -lmx -lmex -lmat -lm -lstdc++
LIBS += -Wl,--start-group $(MKLLIB) -Wl,--end-group
#$(MKLROOT)/../compiler/lib/intel64/libimf.a $(MKLROOT)/../compiler/lib/intel64/libirc.a
LIBS += -lpthread 
else
MATLAB = /share/software/matlab-r2014a
CC = g++
INC = -I$(MATLAB)/extern/include -I$(MATLAB)/simulink/include

OPTS = -D_GNU_SOURCE -DMATLAB_MEX_FILE -ansi -fexceptions -fPIC -fno-omit-frame-pointer -pthread -fopenmp -O3 -DNDEBUG -std=c++11

LIBS = -pthread -Wl,--no-undefined -fopenmp -Wall -shared -O3 -Wl,--version-script,$(MATLAB)/extern/lib/glnxa64/mexFunction.map
LIBS += -lmwlapack -lmwblas
LIBS += -Wl,-rpath-link,$(MATLAB)/bin/glnxa64 -L$(MATLAB)/bin/glnxa64 -lmx -lmex -lmat -lm -lstdc++
LIBS += -lpthread
endif


#TARGET = train_mf
MEX_SUFFIX = mexa64
#SRC = $(TARGET).cpp
#OBJS = $(SRC:.cpp=.o)
OBJS = ../mf.o ../bilinear.o ../multilabel.o ../multiple_linear.o ../smat.o ../dmat.o ../tron.o ../zlib_util.o ../zlib/libz.a

.PHONY: all new clean
.SUFFIXES: .cpp .o

#.cpp.o:
#	$(CC) $(OPTS) $(INC) -c $< -o $@


all: smat_write.$(MEX_SUFFIX) train_mf.$(MEX_SUFFIX) train_ml.$(MEX_SUFFIX)
#all: $(OBJS)
#	$(CC) $(OBJS) $(IMC_OBJS) $(OPTS) -o $(TARGET).$(MEX_SUFFIX) $(LIBS)
#	@echo Binary created!! $(TARGET).$(MEX_SUFFIX)

smat_write.$(MEX_SUFFIX): smat_write.o smat_write.cpp
	$(CC) smat_write.o $(OBJS) -o smat_write.$(MEX_SUFFIX) $(LIBS)
	rm smat_write.o

train_mf.$(MEX_SUFFIX): train_mf.o train_mf.cpp
	$(CC) train_mf.o $(OBJS) -o train_mf.$(MEX_SUFFIX) $(LIBS)
	rm train_mf.o

train_ml.$(MEX_SUFFIX): train_ml.o train_ml.cpp
	$(CC) train_ml.o $(OBJS) -o train_ml.$(MEX_SUFFIX) $(LIBS)
	rm train_ml.o

smat_write.o: smat_write.cpp
	$(CC) $(OPTS) $(INC) -c smat_write.cpp -o smat_write.o

train_mf.o: train_mf.cpp
	$(CC) $(OPTS) $(INC) -c train_mf.cpp -o train_mf.o

train_ml.o: train_ml.cpp
	$(CC) $(OPTS) $(INC) -c train_ml.cpp -o train_ml.o


new: clean all

clean:
	rm -rf *.$(MEX_SUFFIX)


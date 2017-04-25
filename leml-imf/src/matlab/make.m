% This make.m is for MATLAB under Unix/Linux

try
	Type = ver;
	% This part is for OCTAVE
	if(strcmp(Type(1).Name, 'Octave') == 1)
		fprintf('Not supported.\n');
		
	% This part is for MATLAB
	else
		system('make -C ../ lib ');
		% smat_write
		mex -largeArrayDims CFLAGS="\$CFLAGS -fopenmp " LDFLAGS="\$LDFLAGS -fopenmp  -Wall " COMPFLAGS="\$COMPFLAGS -openmp" -cxx smat_write.cpp ../smat.o ../zlib_util.o ../zlib/libz.a
		% train_ml
		mex -largeArrayDims -lmwlapack -lmwblas CFLAGS="\$CFLAGS -fopenmp " LDFLAGS="\$LDFLAGS -fopenmp  -Wall " COMPFLAGS="\$COMPFLAGS -openmp" -cxx train_ml.cpp ../multilabel.o ../bilinear.o ../multiple_linear.o ../smat.o ../dmat.o ../tron.o ../zlib_util.o ../zlib/libz.a
		% train_mf
		mex -largeArrayDims -lmwlapack -lmwblas CFLAGS="\$CFLAGS -fopenmp " LDFLAGS="\$LDFLAGS -fopenmp  -Wall " COMPFLAGS="\$COMPFLAGS -openmp" -cxx train_mf.cpp ../mf.o ../bilinear.o ../multiple_linear.o ../smat.o ../dmat.o ../tron.o ../zlib_util.o ../zlib/libz.a

	end
catch ME
	fprintf('%s\n',getReport(ME));
	fprintf('If make.m failes, please check README about detailed instructions.\n');
end

.PHONY: all test-imf clean new

new: clean all

all: train-ml matlab python test-imf


train-ml: 
	make -C src lib

matlab:
	mkdir -p matlab; cd src/matlab; matlab -nodisplay -nosplash -nojvm -r "make; exit"; cp *mex* ../../matlab/
	# mkdir -p matlab; make -C src/matlab; cp src/matlab/*mex* matlab/

python:
	make -C src/python

test-imf:
	make -C test-imf


zipall:
	make clean all
	cd ../; zip -r leml-imf.zip leml-imf/matlab leml-imf/README \
		leml-imf/datasets leml-imf/COPYRIGHT leml-imf/train-ml

zipsrc:
	make clean
	cd ../; rm -f leml-imf-src.zip; zip -r leml-imf-src.zip leml-imf/


clean:
	rm -rf train-ml matlab
	make -C src clean
	make -C test-imf clean
	make -C src/matlab clean
	make -C src/python clean



main: vec3.h main.cu
	nvcc -gencode arch=compute_86,code=sm_86 \
	     -Xcompiler -Wall,-g,-O3 -Xptxas -O3 -o $@ main.cu

main: vec3.h ray.h sphere.h hitable.h main.cu
	#nvcc -gencode arch=compute_86,code=sm_86 \
	nvcc -Xcompiler -Wall,-g,-O3 -Xptxas -O3 -o $@ main.cu

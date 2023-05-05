build:
	/usr/local/cuda/bin/nvcc main.cu -о out

all:
	/usr/local/cuda/bin/nvcc main.cu -о out
	./out
	rm out
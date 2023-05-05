build:
	/usr/local/cuda/bin/nvcc main.cu -o out

all:
	/usr/local/cuda/bin/nvcc main.cu -o out
	./out
	rm out
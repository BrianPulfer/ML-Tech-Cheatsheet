NVCC = nvcc

all: build

build: matmul

matmul.o:matmul.cu
	$(NVCC) -G -g -o matmul.o -c matmul.cu

matmul: matmul.o
	$(NVCC) -G -g -o matmul matmul.o

time:
	time ./matmul

clean:
	rm -rf *.o matmul

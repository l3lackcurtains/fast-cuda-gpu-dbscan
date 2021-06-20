NVCC= nvcc
CUDAFLAGS= -O3 -std=c++14
CFLAGS= -c

main: main.o indexing.o dbscan.o
	$(NVCC) $(CUDAFLAGS) $^ -o main.exe

test: test.o indexing.o dbscan.o
	$(NVCC) $(CUDAFLAGS) $^ -o test.exe

indexing.o: indexing.cu common.h indexing.h
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) indexing.cu

dbscan.o: dbscan.cu common.h indexing.h dbscan.h
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) dbscan.cu

main.o: main.cu common.h indexing.h dbscan.h
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) main.cu

test.o: test.cu common.h indexing.h dbscan.h
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) test.cu

clean:	
	rm -rf *.o *.exe
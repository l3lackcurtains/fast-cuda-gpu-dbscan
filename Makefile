NVCC= nvcc
CUDAFLAGS= -O3 -std=c++11
CFLAGS= -c

main: main.o indexing.o dbscan.o
	$(NVCC) $(CUDAFLAGS) $^ -o main.exe

indexing.o: indexing.cu common.h indexing.h
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) indexing.cu

dbscan.o: dbscan.cu common.h indexing.h dbscan.h
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) dbscan.cu

main.o: main.cu common.h indexing.h dbscan.h
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) main.cu

clean:	
	rm -rf *.o *.exe
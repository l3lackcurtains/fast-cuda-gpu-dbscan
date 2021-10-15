NVCC= nvcc
CUDAFLAGS= -O3 -std=c++14
CFLAGS= -c

main: main.o makeGraph.o read_dataset.o breadthFirstSearch.o report.o
	$(NVCC) $(CUDAFLAGS) $^ -o ./build/main.exe

makeGraph.o: makeGraph.cu makeGraph.h
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) makeGraph.cu

read_dataset.o: read_dataset.cu read_dataset.h
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) read_dataset.cu

breadthFirstSearch.o: breadthFirstSearch.cu breadthFirstSearch.h
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) breadthFirstSearch.cu

report.o: report.cu report.h
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) report.cu

main.o: main.cu makeGraph.h read_dataset.h breadthFirstSearch.h report.h
	$(NVCC) $(CUDAFLAGS) $(CFLAGS) main.cu

clean:	
	rm -rf ./build/*
	rm -rf *.csv *.o *.exe

dbscan: dbscan.o read_dataset.o report.o
	g++ -fopenmp dbscan.o read_dataset.o report.o -o dbscan  -std=c++14

dbscan.o: dbscan.cpp
	g++ -fopenmp dbscan.cpp -c -O3  -std=c++14

	
report.o: report.c
	g++ -O3 report.c -c -O3  -std=c++14
	
read_dataset.o: read_dataset.c
	g++ read_dataset.c -c -O3  -std=c++14

all: dbscan

clean:
	rm ./*.o ./dbscan ./*.csv
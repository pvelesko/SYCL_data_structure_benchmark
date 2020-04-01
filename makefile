all: intel

intel: *.cpp *.hpp
	icpc -std=c++14 -O3 -qopenmp -qopt-report=5 *.cpp -o intel  -g -xMIC-AVX512

clean:
	rm -f *.o intel *.optrpt

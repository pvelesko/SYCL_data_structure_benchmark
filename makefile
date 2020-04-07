all: intel

intel: main.cpp ComplexSoA.hpp Benchmark.hpp DataStructures.hpp
	icpc -qopenmp -std=c++14 -O2 -o intel main.cpp -g  -xhost

intel-sycl: main.cpp ComplexSoA.hpp Benchmark.hpp DataStructures.hpp
	icpx --intel $(GCCTOOLCHAIN) -fiopenmp -std=c++14 -O2 -fsycl  -o intel main.cpp -g -march=knl -DUSESYCL

cuda: main.cpp ComplexSoA.hpp Benchmark.hpp
	clang++ $(GCCTOOLCHAIN) -L$(IOMP5) -std=c++14 -O3 -fsycl -fopenmp=libiomp5  -fsycl-targets=nvptx64-nvidia-cuda-sycldevice  -o intel main.cpp -g

test: test.cpp *.hpp
	#icpx -fsycl -std=c++14 ./test.cpp -o test -I./ -g -stdlib=libc++ --gcc-toolchain=/soft/compilers/gcc/7.4.0/linux-rhel7-x86_64 /soft/compilers/gcc/7.4.0/linux-rhel7-x86_64/lib64/libstdc++.so
	dpcpp -std=c++14 ./test.cpp -o test -I./ -g

clean:
	rm -f *.o intel *.optrpt ./test

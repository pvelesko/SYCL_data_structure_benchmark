all: intel

intel: *.cpp *.hpp
	icpx -std=c++14 -O3 -qopenmp -qopt-report=5 *.cpp -o intel  -g -xhost

test: test.cpp *.hpp
	#icpx -fsycl -std=c++14 ./test.cpp -o test -I./ -g -stdlib=libc++ --gcc-toolchain=/soft/compilers/gcc/7.4.0/linux-rhel7-x86_64 /soft/compilers/gcc/7.4.0/linux-rhel7-x86_64/lib64/libstdc++.so
	dpcpp -std=c++14 ./test.cpp -o test -I./ -g

clean:
	rm -f *.o intel *.optrpt ./test

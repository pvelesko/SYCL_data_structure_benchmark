#define INTYPE double
#define RTYPE std::complex<INTYPE>
#define CACHELINE 64
#define MEGA 1000000.f
#define GIGA 1000000000.f
#define CRINTPTR const int* __restrict__
#define CRINTYPEPTR const INTYPE* __restrict__
#ifndef DEBUG
#define DEBUG 0
#endif

int M, N;
float R;
void benchmark_args(int argc, char** argv) {
  if (argc != 4) {
    std::cout << "wrong number of arguments. Expecting 3:" << std::endl;
    std::cout << "int   N - vector length" << std::endl;
    std::cout << "int   M - outer loop to run each compute function" << std::endl;
    std::cout << "float R - % ratio of cache line to fill with sequential elements" << std::endl;
    exit(1);
  }

  N = atoi(argv[1]); // vector length
  M = atoi(argv[2]); // outer loop
  R = atof(argv[3]); // ratio
  float size = ((2 * 2 * sizeof(INTYPE)) + (2 * sizeof(int))) * N / MEGA;
  std::cout << "Using N(vector length) = " << N << std::endl;
  std::cout << "Using M(outer loop) = " << M << std::endl;
  std::cout << "Using R(ratio) = " << R << std::endl;
  std::cout << "Footprint = " << size << " MB" << std::endl;
}

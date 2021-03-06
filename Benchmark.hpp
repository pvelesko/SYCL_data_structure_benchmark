#define real_type double
#define RTYPE std::complex<real_type>
#define L3CACHE 32 // Size of L3 Cache in MB
#define HOTCACHE 0 // Should cache be warmed prior to timing
#define CACHELINE 64
#define MEGA 1000000.f
#define GIGA 1000000000.f
#define CRIPTR const int* __restrict__
#define CRRPTR const real_type* __restrict__
#ifndef DEBUG
#define DEBUG 0
#endif

int M, N, num_t;
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
  float size = ((2 * 2 * sizeof(real_type)) + (2 * sizeof(int))) * N / MEGA;
  std::cout << "Using N(vector length) = " << N << std::endl;
  std::cout << "Using M(outer loop) = " << M << std::endl;
  std::cout << "Using R(ratio) = " << R << std::endl;
  std::cout << "Footprint = " << size << " MB" << std::endl;
}

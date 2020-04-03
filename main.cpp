#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <set>
#include <random>
#include <cmath>
#include "ComplexSoA.hpp"
#include "SyclUtil.hpp"
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

std::random_device rd;
std::mt19937 mt(rd());

void check(RTYPE refpsi, RTYPE psi) {
  if (abs(psi - refpsi) > 0.01f) {
    std::cout << "fail" << std::endl;
    std::cout << "Ref: " << refpsi << std::endl;
    std::cout << "Got: " << psi << std::endl;
  }
  else
    return;
    //std::cout << "pass" << std::endl;
};
int pick_vec(std::vector<int> & v) {
  /*
   */
  const int n = v.size();
  if (n == 0) {
    std::cout << "Crap" << std::endl;
    exit(1);
  }
  std::uniform_real_distribution<double> dist(0, n-1);
  int r = dist(mt); // random index in vector
  int ret = v[r]; // get the value
  v.erase(v.begin() + r); // remove returned element
  return ret;
}
void fill_index(const int n, int* c, const int cache_line_bytes, const float ratio) {
  /* Create indirection vector: values[indices[i]]
   * Params:
   *   n - size of the of the 'indices' vector
   *   c - pointer to 'indices' vector/array
   *   cache_line_bytes - size of a cache line for target architecture
   *   ratio - % of subsequent elements in a cache line 
   *           40% would create indirection vector [0,1,2,3,5,7,6,9,8,4]
   *                     first 4/10 elements are sequential, the rest are filled randomly
   *           This simulates the sortedness of indirection
   */
  int cache_line = cache_line_bytes / sizeof(int); // how many index elements fit on one cache line
  int i, j;
  int num_cache_lines = n/cache_line; // # of full cache lines
  int remainder = n % cache_line;  // remaining elements
  int r = remainder > 0 ? 1 : 0; // add a cache line if there's a remainder
  int num_stride = cache_line * ratio; // filled with stride 1 (sequential)
  if (num_stride == 0) num_stride++; // why cna't I use min?
  int num_rand = cache_line - num_stride; // fillled randomly

  // create a vector of leftover indices
  // leftovers from each cache line + randoms left in last cache line
  int k = 0;
  int num_leftovers = num_cache_lines * num_rand + (remainder % num_stride);
  std::vector<int> leftover(num_leftovers);
  for (i = 0; i < num_cache_lines + r; i++) {
    for (j = 0; j < num_rand; j++) {
      int fill = i * cache_line + num_rand + j;
      if (fill < n) {
        leftover[j + num_rand * i] = fill;
//        std::cout << "rem[" << j + num_rand * i << "]=" << fill << std::endl;
      }
    }
  }
  
  

  for (i = 0; i < num_cache_lines + r; i++) {
    for (j = 0; j < num_stride; j++) {
      if (i * cache_line + j < n)
        c[i * cache_line + j] = i * cache_line + j;
    }
    for (j = 0; j < num_rand; j++) {
      // TODO bug here
      if (leftover.size() > 0 and i * cache_line + num_stride + j < n)
        c[i * cache_line + num_stride + j] = pick_vec(leftover);
    }
  }

//  for (i = 0; i < n; i++)
//    std::cout << "c[" << i << "] = " << c[i] << std::endl;
}
RTYPE calc0(const int N, std::vector<RTYPE> & detValues0, std::vector<RTYPE> & detValues1, CRINTPTR det0, CRINTPTR det1) {
  RTYPE psi = 0;
  for (int i = 0; i < N; i++)
    psi += detValues0[det0[i]] * detValues1[det1[i]];
  return psi;
}
RTYPE calc1(const int N, const std::vector<RTYPE> & detValues0, const std::vector<RTYPE> & detValues1, CRINTPTR det0, CRINTPTR det1) {
  INTYPE psi_r = 0, psi_i = 0;
  RTYPE psi = 0;
  for (int i = 0; i < N; i++) 
  {
    psi_r += detValues0[det0[i]].real() * detValues1[det1[i]].real() - detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
    psi_i += detValues0[det0[i]].real() * detValues1[det1[i]].real() + detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
  }
  psi = std::complex<INTYPE>(psi_r, psi_i);
  return psi;
}
RTYPE calc2(const int N, const std::vector<RTYPE> & detValues0, const std::vector<RTYPE> & detValues1, CRINTPTR det0, CRINTPTR det1) {
  INTYPE psi_r = 0, psi_i = 0;
  RTYPE psi = 0;
#pragma omp parallel for simd reduction(+:psi_r, psi_i) 
  for (int i = 0; i < N; i++) 
  {
    psi_r += detValues0[det0[i]].real() * detValues1[det1[i]].real() - detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
    psi_i += detValues0[det0[i]].real() * detValues1[det1[i]].real() + detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
  }
  psi = std::complex<INTYPE>(psi_r, psi_i);
  return psi;
}
RTYPE calc3(const int N, ComplexSoA & mydetValues0, ComplexSoA & mydetValues1, CRINTPTR det0, CRINTPTR det1) {
  RTYPE psi = 0;
  INTYPE psi_r = 0;
  INTYPE psi_i = 0;
#pragma omp parallel for simd reduction(+:psi_r, psi_i) 
  for (int i = 0; i < N; i++) 
  {
    psi_r += mydetValues0.real(det0[i]) * mydetValues1.real(det1[i]) - mydetValues0.imag(det0[i]) * mydetValues1.imag(det1[i]);
    psi_i += mydetValues0.real(det0[i]) * mydetValues1.real(det1[i]) + mydetValues0.imag(det0[i]) * mydetValues1.imag(det1[i]);
  }
  psi = std::complex<INTYPE>(psi_r, psi_i);
  return psi;
}
RTYPE calc4(const int N, CRINTYPEPTR realdetValues0, CRINTYPEPTR realdetValues1, CRINTYPEPTR imagdetValues0, CRINTYPEPTR imagdetValues1, CRINTPTR det0, CRINTPTR det1) {
  RTYPE psi = 0;
  INTYPE psi_r = 0;
  INTYPE psi_i = 0;
#pragma omp parallel for simd reduction(+:psi_r, psi_i) 
  for (int i = 0; i < N; i++) 
  {
    psi_r += realdetValues0[det0[i]] * realdetValues1[det1[i]] - imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
    psi_i += realdetValues0[det0[i]] * realdetValues1[det1[i]] + imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
  }
  psi = std::complex<INTYPE>(psi_r, psi_i);
  return psi;
}

double t0, t1, t2, t3, t4;
#if 0
inline void run_tests() {
  t0 = timer.timeit();
  for (int i = 0; i < M; i++)
    #pragma noinline
    psiref = calc0(N, detValues0, detValues1, det0, det1);
  t0 = timer.timeit();

  t1 = timer.timeit();
  for (int i = 0; i < M; i++)
    #pragma noinline
    psi = calc1(N, detValues0, detValues1, det0, det1);
  t1 = timer.timeit();
  check(psiref, psi);

  t2 = timer.timeit();
  for (int i = 0; i < M; i++)
    #pragma noinline
    psi = calc2(N, detValues0, detValues1, det0, det1);
  t2 = timer.timeit();
  check(psiref, psi);

  t3 = timer.timeit();
  for (int i = 0; i < M; i++)
    #pragma noinline
    psi = calc3(N, mydetValues0, mydetValues1, det0, det1);
  t3 = timer.timeit();
  check(psiref, psi);

  t4 = timer.timeit();
  for (int i = 0; i < M; i++)
    #pragma noinline
    psi = calc4(N, realdetValues0, realdetValues1, imagdetValues0, imagdetValues1, det0, det1);
  t4 = timer.timeit();
  check(psiref, psi);

}
#endif
int main(int argc, char** argv) {
  init();
  int num_t;

//  std::mt19937 mt(rd());
  const int N = atoi(argv[1]);
  const int M = atoi(argv[2]);
  const float R = atof(argv[3]);
  float size = ((2 * 2 * sizeof(INTYPE)) + (2 * sizeof(int))) * N / MEGA;
  std::cout << "Using N = " << N << std::endl;
  std::cout << "Using M(outer loop) = " << M << std::endl;
  std::cout << "Footprint = " << size << " MB" << std::endl;
  std::cout << "Using fill = " << R << std::endl;

  int* det0 = static_cast<int*>(malloc(sizeof(int) * N));
  int* det1 = static_cast<int*>(malloc(sizeof(int) * N));
  std::vector<RTYPE>detValues0(N, std::complex<INTYPE>(1, 1));
  std::vector<RTYPE>detValues1(N, std::complex<INTYPE>(1, 1));
  ComplexSoA mydetValues0(N);
  ComplexSoA mydetValues1(N);
  INTYPE* realdetValues0 = static_cast<INTYPE*>(malloc(sizeof(INTYPE) * N));
  INTYPE* imagdetValues0 = static_cast<INTYPE*>(malloc(sizeof(INTYPE) * N));
  INTYPE* realdetValues1 = static_cast<INTYPE*>(malloc(sizeof(INTYPE) * N));
  INTYPE* imagdetValues1 = static_cast<INTYPE*>(malloc(sizeof(INTYPE) * N));
  RTYPE psi, psiref;

  for (int i = 0; i < N; i++) {
    mydetValues0._real[i] = detValues0[i].real();
    mydetValues1._real[i] = detValues1[i].real();
    mydetValues0._imag[i] = detValues0[i].imag();
    mydetValues1._imag[i] = detValues1[i].imag();
    realdetValues0[i] = detValues0[i].real();
    realdetValues1[i] = detValues1[i].real();
    imagdetValues0[i] = detValues0[i].imag();
    imagdetValues1[i] = detValues1[i].imag();
  }

  timer.timeit("Geneate indirection vector");
  fill_index(N, det0, CACHELINE , R);
  fill_index(N, det1, CACHELINE , R);
  timer.timeit("Geneate indirection vector");



  std::cout << "-------------- RESULT -------------------" << std::endl;
  std::cout << "OpenMP Threads: " << num_t << std::endl;
  //std::cout << std::left << std::setprecision(3) << std::setw(10) << t0    << " Runtime" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0 << " Test0 std::complex" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t1 << " Test1 Real/Imag" << std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t2 << " Test2 Real/Imag SIMD HT" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t3 << " Test3 std::complexSoA" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t4 << " Test4 std::complex Arrays" <<  std::endl;

  //free(det0);
  //free(det1);
  ////free(detValues0);
  ////free(detValues1);
  ////~mydetValues0();
  ////~mydetValues1();
  //free(realdetValues0);
  //free(imagdetValues0);
  //free(realdetValues1);
  //free(imagdetValues1);


  return 0;
}

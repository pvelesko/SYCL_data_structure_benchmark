#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <set>
#include <random>
#include <cmath>
#include <algorithm>
#include "omp.h"
#include "ComplexSoA.hpp"
#include "SyclUtil.hpp"
#include "Benchmark.hpp"

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
void generate_indirection_array(const int n, int* c, const float ratio) {
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
  int num_elements_in_cache_line = CACHELINE / sizeof(int);
  int line_fill_sequential = R * (float)num_elements_in_cache_line;
  int line_fill_random = num_elements_in_cache_line - line_fill_sequential;

  std::vector<int> randos{};
  // initialize
  for (int i = 0; i < n; i++)
    c[i] = -1;

  int idx = 0;
  while(idx != n) {
    for(int i = 0; i < line_fill_sequential; i++) {
      if(idx != n) {
        c[idx] = idx;
        idx++;
      }
    }
    for(int i = 0; i < line_fill_random; i++) {
      if(idx != n) {
        randos.push_back(idx);
        idx++;
      }
    }
  }

  //shuffle leftover indices
 std::random_shuffle(randos.begin(), randos.end());

 // fill the rest
 idx = 0;
 for (int i = 0; i < n; i++) {
   if (c[i] == -1) {
     c[i] = randos[idx];
     idx++;
   }
 }
}
RTYPE calc0(const int N, RTYPE* detValues0, RTYPE* detValues1, CRINTPTR det0, CRINTPTR det1) {
  RTYPE psi = 0;
  for (int i = 0; i < N; i++)
    psi += detValues0[det0[i]] * detValues1[det1[i]];
  return psi;
}
RTYPE calc1(const int N, RTYPE* detValues0, RTYPE* detValues1, CRINTPTR det0, CRINTPTR det1) {
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
RTYPE calc2(const int N, RTYPE* detValues0, RTYPE* detValues1, CRINTPTR det0, CRINTPTR det1) {
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
RTYPE calc3(const int N, ComplexSoA* mydetValues0, ComplexSoA* mydetValues1, CRINTPTR det0, CRINTPTR det1) {
  RTYPE psi = 0;
  INTYPE psi_r = 0;
  INTYPE psi_i = 0;
#pragma omp parallel for simd reduction(+:psi_r, psi_i) 
  for (int i = 0; i < N; i++) 
  {
    psi_r += mydetValues0->real(det0[i]) * mydetValues1->real(det1[i]) - mydetValues0->imag(det0[i]) * mydetValues1->imag(det1[i]);
    psi_i += mydetValues0->real(det0[i]) * mydetValues1->real(det1[i]) + mydetValues0->imag(det0[i]) * mydetValues1->imag(det1[i]);
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

RTYPE calc0_sycl(const int N, RTYPE* detValues0, RTYPE* detValues1, CRINTPTR det0, CRINTPTR det1) {
  RTYPE psi = 0;
  for (int i = 0; i < N; i++)
    psi += detValues0[det0[i]] * detValues1[det1[i]];
  return psi;
}
double t0, t1, t2, t3, t4;
RTYPE psiref;

template<class Lambda, class... Args>
double bench(Lambda lam, Args... args) {
  auto t = timer.timeit();
  auto psi = psiref;;
  for (int i = 0; i < M; i++)
    psi = lam(args...);
  t = timer.timeit();
  check(psiref, psi);
  return t; 
}

int main(int argc, char** argv) {
  benchmark_args(argc, argv);
  init();
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
  generate_indirection_array(N, det0, R);
  generate_indirection_array(N, det1, R);
  timer.timeit("Geneate indirection vector");


  psiref = calc0(N, detValues0.data(), detValues1.data(), det0, det1);
  t0 = bench(calc0, N, detValues0.data(), detValues1.data(), det0, det1);
  t1 = bench(calc1, N, detValues0.data(), detValues1.data(), det0, det1);
  t2 = bench(calc2, N, detValues0.data(), detValues1.data(), det0, det1);
  t3 = bench(calc3, N, &mydetValues0, &mydetValues1, det0, det1);
  t4 = bench(calc4, N, realdetValues0, realdetValues1, imagdetValues0, imagdetValues1, det0, det1);

  int num_t;
  #pragma omp master
  #pragma omp parallel
  {
    num_t = omp_get_num_threads();
  }
  std::cout << "-------------- RESULT -------------------" << std::endl;
  std::cout << "OpenMP Threads: " << num_t << std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0    << "Runtime" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t0 << "Speedup Test0 std::complex" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t1 << "Speedup Test1 Real/Imag" << std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t2 << "Speedup Test2 Real/Imag SIMD HT" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t3 << "Speedup Test3 std::complexSoA" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t4 << "Speedup Test4 std::complex Arrays" <<  std::endl;

  return 0;
}

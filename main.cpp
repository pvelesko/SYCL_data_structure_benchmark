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
#include "DataStructures.hpp"

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
RTYPE calc0(const int N, RTYPE* detValues0, RTYPE* detValues1, CRIPTR det0, CRIPTR det1) {
  RTYPE psi = 0;
  for (int i = 0; i < N; i++)
    psi += detValues0[det0[i]] * detValues1[det1[i]];
  return psi;
}
RTYPE calc1(const int N, RTYPE* detValues0, RTYPE* detValues1, CRIPTR det0, CRIPTR det1) {
  real_type psi_r = 0, psi_i = 0;
  RTYPE psi = 0;
  for (int i = 0; i < N; i++) 
  {
    psi_r += detValues0[det0[i]].real() * detValues1[det1[i]].real() - detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
    psi_i += detValues0[det0[i]].real() * detValues1[det1[i]].real() + detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
  }
  psi = std::complex<real_type>(psi_r, psi_i);
  return psi;
}
RTYPE calc2(const int N, RTYPE* detValues0, RTYPE* detValues1, CRIPTR det0, CRIPTR det1) {
  real_type psi_r = 0, psi_i = 0;
  RTYPE psi = 0;
#pragma omp parallel for simd reduction(+:psi_r, psi_i) 
  for (int i = 0; i < N; i++) 
  {
    psi_r += detValues0[det0[i]].real() * detValues1[det1[i]].real() - detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
    psi_i += detValues0[det0[i]].real() * detValues1[det1[i]].real() + detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
  }
  psi = std::complex<real_type>(psi_r, psi_i);
  return psi;
}
RTYPE calc3(const int N, ComplexSoA* mydetValues0, ComplexSoA* mydetValues1, CRIPTR det0, CRIPTR det1) {
  RTYPE psi = 0;
  real_type psi_r = 0;
  real_type psi_i = 0;
#pragma omp parallel for simd reduction(+:psi_r, psi_i) 
  for (int i = 0; i < N; i++) 
  {
    psi_r += mydetValues0->real(det0[i]) * mydetValues1->real(det1[i]) - mydetValues0->imag(det0[i]) * mydetValues1->imag(det1[i]);
    psi_i += mydetValues0->real(det0[i]) * mydetValues1->real(det1[i]) + mydetValues0->imag(det0[i]) * mydetValues1->imag(det1[i]);
  }
  psi = std::complex<real_type>(psi_r, psi_i);
  return psi;
}
RTYPE calc4(const int N, CRRPTR realdetValues0, CRRPTR realdetValues1, CRRPTR imagdetValues0, CRRPTR imagdetValues1, CRIPTR det0, CRIPTR det1) {
  RTYPE psi = 0;
  real_type psi_r = 0;
  real_type psi_i = 0;
#pragma omp parallel for simd reduction(+:psi_r, psi_i) 
  for (int i = 0; i < N; i++) 
  {
    psi_r += realdetValues0[det0[i]] * realdetValues1[det1[i]] - imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
    psi_i += realdetValues0[det0[i]] * realdetValues1[det1[i]] + imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
  }
  psi = std::complex<real_type>(psi_r, psi_i);
  return psi;
}

RTYPE calc0_sycl(const int N, RTYPE* detValues0, RTYPE* detValues1, CRIPTR det0, CRIPTR det1) {
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
  usm_allocator<char, usm::alloc::shared> usmallocator(q.get_context(), q.get_device());
  std::allocator<char> stdallocator{};

  typedef SoA<decltype(usmallocator), RTYPE*, int*> StdComplexIndSoA;
  typedef SoA<decltype(usmallocator), RTYPE*, RTYPE*, int*> MyComplexIndSoA;

  _voidptr = static_cast<void*>(usmallocator.allocate(sizeof(StdComplexIndSoA)));
  StdComplexIndSoA* cSoA0 = new (_voidptr) StdComplexIndSoA(usmallocator, N);
  _voidptr = static_cast<void*>(usmallocator.allocate(sizeof(StdComplexIndSoA)));
  StdComplexIndSoA* cSoA1 = new (_voidptr) StdComplexIndSoA(usmallocator, N);

  timer.timeit("Geneate indirection vector");
  generate_indirection_array(N, cSoA0->data<1>(), R);
  generate_indirection_array(N, cSoA1->data<1>(), R);
  timer.timeit("Geneate indirection vector");


  psiref = calc0(N, cSoA0->data<0>(), cSoA1->data<0>(), cSoA0->data<1>(), cSoA1->data<1>());
  t0 = bench(calc0, N, cSoA0->data<0>(), cSoA1->data<0>(), cSoA0->data<1>(), cSoA1->data<1>());
  t1 = bench(calc1, N, cSoA0->data<0>(), cSoA1->data<0>(), cSoA0->data<1>(), cSoA1->data<1>());
  t2 = bench(calc2, N, cSoA0->data<0>(), cSoA1->data<0>(), cSoA0->data<1>(), cSoA1->data<1>());
//  t3 = bench(calc3, N, &mydetValues0, &mydetValues1, det0, det1);
//  t4 = bench(calc4, N, realdetValues0, realdetValues1, imagdetValues0, imagdetValues1, det0, det1);

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

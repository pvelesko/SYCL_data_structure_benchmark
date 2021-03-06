#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <complex>
#include <algorithm>
#include "omp.h"
#include "ComplexSoA.hpp"
#include "SyclUtil.hpp"
#include "Benchmark.hpp"
#include "DataStructures.hpp"

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
RTYPE calc1_simd(const int N, RTYPE* detValues0, RTYPE* detValues1, CRIPTR det0, CRIPTR det1) {
  real_type psi_r = 0, psi_i = 0;
  RTYPE psi = 0;
#pragma omp simd reduction(+:psi_r, psi_i) 
  for (int i = 0; i < N; i++) 
  {
    psi_r += detValues0[det0[i]].real() * detValues1[det1[i]].real() - detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
    psi_i += detValues0[det0[i]].real() * detValues1[det1[i]].real() + detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
  }
  psi = std::complex<real_type>(psi_r, psi_i);
  return psi;
}
RTYPE calc1_simd_ht(const int N, RTYPE* detValues0, RTYPE* detValues1, CRIPTR det0, CRIPTR det1) {
  real_type psi_r = 0, psi_i = 0;
  RTYPE psi = 0;
#pragma omp parallel for simd reduction(+:psi_r, psi_i) num_threads(num_t)
  for (int i = 0; i < N; i++) 
  {
    psi_r += detValues0[det0[i]].real() * detValues1[det1[i]].real() - detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
    psi_i += detValues0[det0[i]].real() * detValues1[det1[i]].real() + detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
  }
  psi = std::complex<real_type>(psi_r, psi_i);
  return psi;
}
RTYPE calc1_simd_ht_schedule(const int N, RTYPE* detValues0, RTYPE* detValues1, CRIPTR det0, CRIPTR det1) {
  real_type psi_r = 0, psi_i = 0;
  RTYPE psi = 0;
#pragma omp parallel for simd reduction(+:psi_r, psi_i) schedule(static, 1) num_threads(num_t)
  for (int i = 0; i < N; i++) 
  {
    psi_r += detValues0[det0[i]].real() * detValues1[det1[i]].real() - detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
    psi_i += detValues0[det0[i]].real() * detValues1[det1[i]].real() + detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
  }
  psi = std::complex<real_type>(psi_r, psi_i);
  return psi;
}
RTYPE calc1_ht_schedule(const int N, RTYPE* detValues0, RTYPE* detValues1, CRIPTR det0, CRIPTR det1) {
  real_type psi_r = 0, psi_i = 0;
  RTYPE psi = 0;
#pragma omp parallel for schedule(static, 1) num_threads(num_t)
  for (int i = 0; i < N; i++) 
  {
    psi_r += detValues0[det0[i]].real() * detValues1[det1[i]].real() - detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
    psi_i += detValues0[det0[i]].real() * detValues1[det1[i]].real() + detValues0[det0[i]].imag() * detValues1[det1[i]].imag();
  }
  psi = std::complex<real_type>(psi_r, psi_i);
  return psi;
}

RTYPE calc2(const int N, CRRPTR realdetValues0, CRRPTR realdetValues1, CRRPTR imagdetValues0, CRRPTR imagdetValues1, CRIPTR det0, CRIPTR det1) {
  RTYPE psi = 0;
  real_type psi_r = 0;
  real_type psi_i = 0;
  for (int i = 0; i < N; i++) 
  {
    psi_r += realdetValues0[det0[i]] * realdetValues1[det1[i]] - imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
    psi_i += realdetValues0[det0[i]] * realdetValues1[det1[i]] + imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
  }
  psi = std::complex<real_type>(psi_r, psi_i);
  return psi;
}
RTYPE calc2_simd(const int N, CRRPTR realdetValues0, CRRPTR realdetValues1, CRRPTR imagdetValues0, CRRPTR imagdetValues1, CRIPTR det0, CRIPTR det1) {
  RTYPE psi = 0;
  real_type psi_r = 0;
  real_type psi_i = 0;
#pragma omp simd reduction(+:psi_r, psi_i) 
  for (int i = 0; i < N; i++) 
  {
    psi_r += realdetValues0[det0[i]] * realdetValues1[det1[i]] - imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
    psi_i += realdetValues0[det0[i]] * realdetValues1[det1[i]] + imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
  }
  psi = std::complex<real_type>(psi_r, psi_i);
  return psi;
}
RTYPE calc2_simd_ht(const int N, CRRPTR realdetValues0, CRRPTR realdetValues1, CRRPTR imagdetValues0, CRRPTR imagdetValues1, CRIPTR det0, CRIPTR det1) {
  RTYPE psi = 0;
  real_type psi_r = 0;
  real_type psi_i = 0;
#pragma omp parallel for simd reduction(+:psi_r, psi_i) num_threads(num_t)
  for (int i = 0; i < N; i++) 
  {
    psi_r += realdetValues0[det0[i]] * realdetValues1[det1[i]] - imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
    psi_i += realdetValues0[det0[i]] * realdetValues1[det1[i]] + imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
  }
  psi = std::complex<real_type>(psi_r, psi_i);
  return psi;
}
RTYPE calc2_simd_ht_schedule(const int N, CRRPTR realdetValues0, CRRPTR realdetValues1, CRRPTR imagdetValues0, CRRPTR imagdetValues1, CRIPTR det0, CRIPTR det1) {
  RTYPE psi = 0;
  real_type psi_r = 0;
  real_type psi_i = 0;
#pragma omp parallel for simd reduction(+:psi_r, psi_i) schedule(static, 1) num_threads(num_t)
  for (int i = 0; i < N; i++) 
  {
    psi_r += realdetValues0[det0[i]] * realdetValues1[det1[i]] - imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
    psi_i += realdetValues0[det0[i]] * realdetValues1[det1[i]] + imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
  }
  psi = std::complex<real_type>(psi_r, psi_i);
  return psi;
}
RTYPE calc2_ht_schedule(const int N, CRRPTR realdetValues0, CRRPTR realdetValues1, CRRPTR imagdetValues0, CRRPTR imagdetValues1, CRIPTR det0, CRIPTR det1) {
  RTYPE psi = 0;
  real_type psi_r = 0;
  real_type psi_i = 0;
#pragma omp parallel for schedule(static, 1) num_threads(num_t)
  for (int i = 0; i < N; i++) 
  {
    psi_r += realdetValues0[det0[i]] * realdetValues1[det1[i]] - imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
    psi_i += realdetValues0[det0[i]] * realdetValues1[det1[i]] + imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
  }
  psi = std::complex<real_type>(psi_r, psi_i);
  return psi;
}

#ifdef USESYCL
RTYPE calc3_sycl(const int N, CRRPTR realdetValues0, CRRPTR realdetValues1, CRRPTR imagdetValues0, CRRPTR imagdetValues1, CRIPTR det0, CRIPTR det1) {
  RTYPE psi = 0;
  real_type psi_r = 0;
  real_type psi_i = 0;
  real_type* tmp0 = static_cast<real_type*>(malloc_shared(N * sizeof(real_type), q));
  real_type* tmp1 = static_cast<real_type*>(malloc_shared(N * sizeof(real_type), q));
  auto e = q.submit([&](handler& cgh) {
    cgh.parallel_for<class calc3_sycl>(range<1>(N), [=](id<1> i) {
      tmp0[i] += realdetValues0[det0[i]] * realdetValues1[det1[i]] - imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
      tmp1[i] += realdetValues0[det0[i]] * realdetValues1[det1[i]] + imagdetValues0[det0[i]] * imagdetValues1[det1[i]];
    }); // par for
  }); // queue
  e.wait();

  // reduce
  for (int i = 0; i < N; i++) {
    psi_r += tmp0[i];
    psi_i += tmp1[i];
  }

  psi = std::complex<real_type>(psi_r, psi_i);
  return psi;
}
#endif
RTYPE psiref;

template<class Lambda, class... Args>
double bench(Lambda lam, Args... args) {
  int num_ints = 10 * L3CACHE / sizeof(int) * 1e6; //clear with 10x sizeof $L3
  volatile int* junk = static_cast<int*>(malloc(num_ints * sizeof(int))); 
  double nett = 0;
  for (int i = 0; i < M; i++) {
    auto psi = psiref; psi = 0;
    // Crear the cache - allocate and read array bigger than L3
    if (HOTCACHE) {
      psi = lam(args...);
    } else {
      for (int i = 0; i < num_ints; i++)
        junk[i] += 1;
    }
    auto t = timer.timeit();
      psi = lam(args...);
    t = timer.timeit();
    check(psiref, psi);
    nett += t;
    }
  nett = nett / M; //get average instance time
  std::free((void*)junk);
  return nett; 
}

int main(int argc, char** argv) {
  benchmark_args(argc, argv);
  init();
  std::ofstream out;
  out.open("data.txt");
  out << "OpenMP Threads,Footprint,Ratio,Test0..." << std::endl;
  float size = ((2 * 2 * sizeof(real_type)) + (2 * sizeof(int))) * N / MEGA;
  #pragma omp parallel
  #pragma omp master
  {
    num_t = omp_get_num_threads();
  }

  typedef SoA<decltype(allocator), RTYPE*, int*> StdComplexIndSoA;
  typedef SoA<decltype(allocator), real_type*, real_type*, int*> MyComplexIndSoA;

  for (num_t = 1; num_t < 5; num_t++) {
    for (N = 2500; N < 2500001; N *= 10) {
      for (R = 0; R < 1; R += 0.1) {

  _voidptr = static_cast<void*>(allocator.allocate(sizeof(StdComplexIndSoA)));
  StdComplexIndSoA* cSoA0 = new (_voidptr) StdComplexIndSoA(allocator, N);
  _voidptr = static_cast<void*>(allocator.allocate(sizeof(StdComplexIndSoA)));
  StdComplexIndSoA* cSoA1 = new (_voidptr) StdComplexIndSoA(allocator, N);

  _voidptr = static_cast<void*>(allocator.allocate(sizeof(MyComplexIndSoA)));
  MyComplexIndSoA* mycSoA0 = new (_voidptr) MyComplexIndSoA(allocator, N);
  _voidptr = static_cast<void*>(allocator.allocate(sizeof(MyComplexIndSoA)));
  MyComplexIndSoA* mycSoA1 = new (_voidptr) MyComplexIndSoA(allocator, N);

  timer.timeit("Geneate indirection vector");
  generate_indirection_array(N, cSoA0->data<1>(), R);
  generate_indirection_array(N, cSoA1->data<1>(), R);
  generate_indirection_array(N, mycSoA0->data<2>(), R);
  generate_indirection_array(N, mycSoA1->data<2>(), R);
  timer.timeit("Geneate indirection vector");



  psiref = calc0(N, cSoA0->data<0>(), cSoA1->data<0>(), cSoA0->data<1>(), cSoA1->data<1>());
  auto t0  = bench(calc0, N, cSoA0->data<0>(), cSoA1->data<0>(), cSoA0->data<1>(), cSoA1->data<1>());

  auto t1  = bench(calc1, N, cSoA0->data<0>(), cSoA1->data<0>(), cSoA0->data<1>(), cSoA1->data<1>());
  auto t2  = bench(calc1_simd, N, cSoA0->data<0>(), cSoA1->data<0>(), cSoA0->data<1>(), cSoA1->data<1>());
  auto t3  = bench(calc1_simd_ht, N, cSoA0->data<0>(), cSoA1->data<0>(), cSoA0->data<1>(), cSoA1->data<1>());
  auto t4  = bench(calc1_simd_ht_schedule, N, cSoA0->data<0>(), cSoA1->data<0>(), cSoA0->data<1>(), cSoA1->data<1>());
  auto t5  = bench(calc1_ht_schedule, N, cSoA0->data<0>(), cSoA1->data<0>(), cSoA0->data<1>(), cSoA1->data<1>());
 
  auto t6  = bench(calc2, N, mycSoA0->data<0>(), mycSoA1->data<0>(), mycSoA0->data<1>(), mycSoA1->data<1>(), mycSoA0->data<2>(), mycSoA1->data<2>());
  auto t7  = bench(calc2_simd, N, mycSoA0->data<0>(), mycSoA1->data<0>(), mycSoA0->data<1>(), mycSoA1->data<1>(), mycSoA0->data<2>(), mycSoA1->data<2>());
  auto t8  = bench(calc2_simd_ht, N, mycSoA0->data<0>(), mycSoA1->data<0>(), mycSoA0->data<1>(), mycSoA1->data<1>(), mycSoA0->data<2>(), mycSoA1->data<2>());
  auto t9  = bench(calc2_simd_ht_schedule, N, mycSoA0->data<0>(), mycSoA1->data<0>(), mycSoA0->data<1>(), mycSoA1->data<1>(), mycSoA0->data<2>(), mycSoA1->data<2>());
  auto t10 = bench(calc2_ht_schedule, N, mycSoA0->data<0>(), mycSoA1->data<0>(), mycSoA0->data<1>(), mycSoA1->data<1>(), mycSoA0->data<2>(), mycSoA1->data<2>());
      //auto t5 = bench(calc3_sycl, N, mycSoA0->data<0>(), mycSoA1->data<0>(), mycSoA0->data<1>(), mycSoA1->data<1>(), mycSoA0->data<2>(), mycSoA1->data<2>());


  out << num_t  << "," << N << "," << R << ",";
  out << t0/t1  << ",";
  out << t0/t2  << ",";
  out << t0/t3  << ",";
  out << t0/t4  << ",";
  out << t0/t5  << ",";
  out << t0/t6  << ",";
  out << t0/t7  << ",";
  out << t0/t8  << ",";
  out << t0/t9  << ",";
  out << t0/t10 << std::endl;

  std::cout << "-------------- RESULT -------------------" << std::endl;
  std::cout << "OpenMP Threads: " << num_t << std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0     << "Runtime Baseline (Test0)" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t0  << "Speedup Test0  std::complex complex operators" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t1  << "Speedup Test1  std::Complex Real/Imag" << std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t2  << "Speedup Test2  std::Complex Real/Imag SIMD" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t3  << "Speedup Test3  std::Complex Real/Imag SIMD HT" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t4  << "Speedup Test4  std::Complex Real/Imag SIMD HT schedule" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t5  << "Speedup Test5  std::Complex Real/Imag HT schedule" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t6  << "Speedup Test6  MyComplex    Real/Imag" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t7  << "Speedup Test7  MyComplex    Real/Imag SIMD" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t8  << "Speedup Test8  MyComplex    Real/Imag SIMD HT" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t9  << "Speedup Test9  MyComplex    Real/Imag SIMD HT schedule" <<  std::endl;
  std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t10 << "Speedup Test10 MyComplex    Real/Imag HT schedule" <<  std::endl;
  //std::cout << std::left << std::setprecision(3) << std::setw(10) << t0/t5 << "Speedup Test5 MyComplex    Real/Imag SIMD HT SYCL" <<  std::endl;

      } // Ratio
    } // size
  } // threads

  return 0;
}

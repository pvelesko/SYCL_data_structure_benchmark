#include "SyclUtil.hpp"
#include "DataStructures.hpp"
#include "CL/sycl.hpp"
#include <complex>
using namespace cl::sycl;

int main(int argc, char** argv) {
  process_args(argc, argv);
  init();
  usm_allocator<char, usm::alloc::shared> usmallocator(q.get_context(), q.get_device());
  std::allocator<char> stdallocator{};
  void* voidptr; // for allocating USM space for top-level classes


  typedef AoS<decltype(usmallocator), Particle<float>> ParticleAoS;
  voidptr = static_cast<void*>(usmallocator.allocate(sizeof(ParticleAoS)));
  ParticleAoS* pAoS = new (voidptr) ParticleAoS(usmallocator, n);
  voidptr = NULL;

  // Dry run
  e = par_for(n, [=](int i) {
    pAoS->data()[i].pos_x = 1;
    pAoS->data()[i].pos_y = 2;
    pAoS->data()[i].pos_z = 3;
  });
  e.wait();
  timer.timeit("noop");
  e = par_for(n, [=](int i) {
  });
  e.wait();
  timer.timeit("noop");

  timer.timeit("pAoS");
  e = par_for(n, [=](int i) {
    pAoS->data()[i].pos_x = 1;
    pAoS->data()[i].pos_y = 2;
    pAoS->data()[i].pos_z = 3;
  });
  e.wait();
  timer.timeit("pAoS");
#ifdef DEBUG
  dump(pAoS->data(), "pAoS");
#endif 
  pAoS->~ParticleAoS();

  typedef SoA<decltype(usmallocator), float*, float*, float*> ParticleSoA;
  voidptr = static_cast<void*>(usmallocator.allocate(sizeof(ParticleSoA)));
  ParticleSoA* pSoA = new (voidptr) ParticleSoA(usmallocator, n);
  voidptr = NULL;
 
  timer.timeit("pSoA");
  e = par_for(n, [=](int i) {
    pSoA->data<0>()[i] = 1;
    pSoA->data<1>()[i] = 2;
    pSoA->data<2>()[i] = 3;
  });
  e.wait();
  timer.timeit("pSoA");
#ifdef DEBUG
  for(int i = 0; i < n; i++) {
    std::cout << "pSoA[" << i << "] = ";
    std::cout << "(" << pSoA->data<0>()[i];
    std::cout << "," << pSoA->data<1>()[i];
    std::cout << "," << pSoA->data<2>()[i];
    std::cout << ")" << std::endl;
  }
#endif 
  pSoA->~ParticleSoA();

  typedef AoS<decltype(usmallocator), std::complex<float>> ComplexAoS;
  voidptr = static_cast<void*>(usmallocator.allocate(sizeof(ComplexAoS)));
  ComplexAoS* complexAoS = new (voidptr) ComplexAoS(usmallocator, n);
  voidptr = NULL;
 
  timer.timeit("complexAoS");
  e = par_for(n, [=](int i) {
    complexAoS->data()[i] = std::complex<float>(-1, 1);
  });
  e.wait();
  timer.timeit("complexAos");
#ifdef DEBUG
  dump(complexAoS->data(), "complexAoS");
#endif
  complexAoS->~ComplexAoS();


  typedef SoA<decltype(usmallocator), float*, float*> ComplexSoA;
  voidptr = static_cast<void*>(usmallocator.allocate(sizeof(ComplexSoA)));
  ComplexSoA* complexSoA = new (voidptr) ComplexSoA(usmallocator, n);
  voidptr = NULL;
 
  timer.timeit("complexSoA");
  e = par_for(n, [=](int i) {
    complexSoA->data<0>()[i] = -1;
    complexSoA->data<1>()[i] = 1;
  });
  e.wait();
  timer.timeit("complexSoA");
#ifdef DEBUG
  for(int i = 0; i < n; i++) {
    std::cout << "complexSoA[" << i << "] = ";
    std::cout << "(" << complexSoA->data<0>()[i];
    std::cout << "," << complexSoA->data<1>()[i];
    std::cout << ")" << std::endl;
  }
#endif 
  complexSoA->~ComplexSoA();

  return 0;
}

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

  typedef AoS<decltype(stdallocator), Particle<float>> ParticleAoS;
  typedef AoS<decltype(usmallocator), Particle<float>> ParticleAoSSycl;
  typedef SoA<decltype(usmallocator), float*, float*, float*> ParticleSoASycl;
  typedef AoS<decltype(usmallocator), Particle<float>> ParticleAoSSycl;

  typedef AoS<decltype(usmallocator), std::complex<float>> ComplexAoSSycl;

  voidptr = static_cast<void*>(usmallocator.allocate(sizeof(ParticleAoSSycl)));
  ParticleAoSSycl* pAoS = new (voidptr) ParticleAoSSycl(usmallocator, n);
  voidptr = NULL;

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
  pAoS->~ParticleAoSSycl();

  voidptr = static_cast<void*>(usmallocator.allocate(sizeof(ParticleSoASycl)));
  ParticleSoASycl* pSoA = new (voidptr) ParticleSoASycl(usmallocator, n);
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
  pSoA->~ParticleSoASycl();

  voidptr = static_cast<void*>(usmallocator.allocate(sizeof(ComplexAoSSycl)));
  ComplexAoSSycl* complexAoS = new (voidptr) ComplexAoSSycl(usmallocator, n);
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
  complexAoS->~ComplexAoSSycl();


  typedef SoA<decltype(usmallocator), float*, float*> ComplexSoASycl;
  voidptr = static_cast<void*>(usmallocator.allocate(sizeof(ComplexSoASycl)));
  ComplexSoASycl* complexSoA = new (voidptr) ComplexSoASycl(usmallocator, n);
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
  complexSoA->~ComplexSoASycl();

  return 0;
}

#include "SyclUtil.hpp"
#include "DataStructures.hpp"
#include "CL/sycl.hpp"
using namespace cl::sycl;

int main(int argc, char** argv) {
  process_args(argc, argv);
  init();
  usm_allocator<char, usm::alloc::shared> usmallocator(q.get_context(), q.get_device());
  std::allocator<char> stdallocator{};

  typedef AoS<decltype(stdallocator), Particle<float>> ParticleAoS;
  typedef AoS<decltype(usmallocator), Particle<float>> ParticleAoSSycl;
  typedef SoA<decltype(usmallocator), float*, float*, float*> ParticleSoASycl;
  void* voidptr; // for allocating USM space for top-level classes
  event e; // capture event for waiting or profiling

  voidptr = static_cast<void*>(usmallocator.allocate(sizeof(ParticleAoSSycl)));
  ParticleAoSSycl* pAoS = new (voidptr) ParticleAoSSycl(usmallocator, n);
  voidptr = NULL;

  e = par_for(n, [=](int i) {
    pAoS->data()[i].pos_x = 1;
    pAoS->data()[i].pos_y = 2;
    pAoS->data()[i].pos_z = 3;
  });

  e.wait();
  dump(pAoS->data(), "pAoS");

  voidptr = static_cast<void*>(usmallocator.allocate(sizeof(ParticleSoASycl)));
  ParticleSoASycl* pSoA = new (voidptr) ParticleSoASycl(usmallocator, n);
  voidptr = NULL;
 
  e = par_for(n, [=](int i) {
    pSoA->data<0>()[i] = 1;
    pSoA->data<1>()[i] = 2;
    pSoA->data<2>()[i] = 3;
  });
  e.wait();
  for(int i = 0; i < n; i++) {
    std::cout << "pSoA[" << i << "] = ";
    std::cout << "(" << pSoA->data<0>()[i];
    std::cout << "," << pSoA->data<1>()[i];
    std::cout << "," << pSoA->data<2>()[i];
    std::cout << ")" << std::endl;
  }


  return 0;
}

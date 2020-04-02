#include "SyclUtil.hpp"
#include "DataStructures.hpp"
#include "CL/sycl.hpp"
using namespace cl::sycl;

int main(int argc, char** argv) {
  process_args(argc, argv);
  init();
  usm_allocator<char, usm::alloc::shared> usmallocator(q.get_context(), q.get_device());
  std::allocator<char> stdallocator{};

  AoS<decltype(stdallocator), Particle<float>> ParticleAoS(stdallocator, n);
  AoS<decltype(usmallocator), Particle<float>> ParticleAoSSycl(usmallocator, n);

  int* x = static_cast<int*>(malloc_shared(n * sizeof(int), q));
  auto e = par_for(n, [=](int i) { x[i] = 1; } );
  e.wait();
  dump(x, "x");

  SoA<decltype(usmallocator), float*, int*> a(usmallocator, n);
  dump(a.data<0>(), "SoA(0)");
  return 0;
}

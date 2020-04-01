#include "SyclUtil.hpp"
#include "DataStructures.hpp"
#include "CL/sycl.hpp"
using namespace cl::sycl;
int main(int argc, char** argv) {
  process_args(argc, argv);
  init();
  usm_allocator<Particle<float>, usm::alloc::shared> allocator(q.get_context(), q.get_device());

  AoS<Particle<float>> ParticleAoS(n);
  AoS<Particle<float>, decltype(allocator)> ParticleAoSSycl(allocator, n);

  int* x = static_cast<int*>(malloc_shared(n * sizeof(int), q));
  par_for(n, [=](int i) { x[i] = 1; } );
  return 0;
}

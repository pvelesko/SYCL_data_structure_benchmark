#include "SyclUtil.hpp"
#include "DataStructures.hpp"
#include "CL/sycl.hpp"
using namespace cl::sycl;
int main(int argc, char** argv) {
  process_args(argc, argv);
  init();
  usm_allocator<int, usm::alloc::shared> allocator(q.get_context(), q.get_device());
  AoS<Particle<float>> ParticleAoS(n);
  return 0;
}

#include "CL/sycl.hpp"
#include <chrono>
using namespace cl::sycl;
int n;
queue q;
device dev;
context ctx;
event e;

class Timer {
  std::chrono::time_point<std::chrono::high_resolution_clock> t0, t1;
  bool tfinish = false;
  public:
  Timer() {};

  // TODO make it track multiple times based on label
  inline double timeit() {
    if(!tfinish) {
      t0 = std::chrono::high_resolution_clock::now();
      tfinish = !tfinish;
      return -1;
    } else {
      t1 = std::chrono::high_resolution_clock::now();
      tfinish = !tfinish;
      auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
      return (double)duration;
    }
  }

  inline double timeit(const std::string name) {
    double res = timeit();
    if (res > 0) 
      std::cout << name << ": " << res/1000000.0 <<"s" << std::endl;
    return res;
  }

  inline double timeit(const std::string name, const event e) {
    auto start = e.get_profiling_info<info::event_profiling::command_start>();
    auto end = e.get_profiling_info<info::event_profiling::command_end>();
    auto total = end - start;
    std::cout << name << "(event): " << total * 10e-9 <<"s" << std::endl;
    return (double)(total * 10-9);
  }

};

auto exception_handler = [] (cl::sycl::exception_list exceptions) {
  for (std::exception_ptr const& e : exceptions) {
    try {
  std::rethrow_exception(e);
    } catch(cl::sycl::exception const& e) {
  std::cout << "Caught asynchronous SYCL exception:\n"
        << e.what() << std::endl;
    }
  }
};
Timer timer;


inline void init() {
  timer = Timer{};
  std::string env;
  if (std::getenv("SYCL_DEVICE") != NULL) {
    env = std::string(std::getenv("SYCL_DEVICE"));
  } else {
    env = std::string("");
  }
  std::cout << "Using DEVICE = " << env << std::endl;
  property_list proplist{property::queue::enable_profiling()};
  if (!env.compare("gpu") or !env.compare("GPU")) {
    q = cl::sycl::queue(cl::sycl::gpu_selector{}, exception_handler, proplist);
  } else if (!env.compare("cpu") or !env.compare("CPU")) {
    q = cl::sycl::queue(cl::sycl::cpu_selector{}, exception_handler, proplist);
  } else if (!env.compare("host") or !env.compare("HOST")) {
    q = cl::sycl::queue(cl::sycl::host_selector{}, exception_handler, proplist);
  } else {
    q = cl::sycl::queue(cl::sycl::default_selector{}, exception_handler, proplist);
  }
  dev = q.get_device();
  ctx = q.get_context();
  std::cout << "Running on "
            << dev.get_info<info::device::name>()
            << std::endl;
};

inline void process_args(int argc, char** argv) {
  if (argc > 1) {
    n = std::atoi(argv[1]);
  } else {
    n = 3;
  }
  std::cout << "Using N = " << n << std::endl;
}

template<class T>
event par_for(const size_t size, T lam) {
  range<1> r(size);
  event e = q.submit([&](handler& cgh) {
    cgh.parallel_for(r, [=](id<1> idx) {
      lam(idx);
    }); //par for
  });// queue scope
  return e;
}

template<class T>
event sin_task(T lam) {
  event e = q.submit([&](handler& cgh) {
    cgh.single_task([=]() {
      lam();
    }); //par for
  });// queue scope
  return e;
}

template<class T>
inline void dump(T* var, std::string name) {
  for(int i = 0; i < n; i++)
    std::cout << name << "[" << i << "] = " << var[i] << std::endl;
}

template<class T>
inline void dump(T* var, int start, int end, std::string name) {
  for(int i = start; i < end; i++)
    std::cout << name << "[" << i << "] = " << var[i] << std::endl;
}


template<class T>
inline void dump(T var, std::string name) {
  std::cout << name << " = " << var << std::endl;
}

#include <iostream>
#include <tuple>

/* Particle data type to completely define one particle
 */
template<class T>
class Particle {
  public:
  T pos_x, pos_y, pos_z;

  Particle() {};
  Particle(T x, T y, T z) {
    pos_x = x;
    pos_y = y;
    pos_z = z;
  };
};

template<class Allocator, class T>
class AoS {
  Allocator m_allocator;
  std::string m_name;
  size_t m_size;
  T* m_data;
  public:
  //AoS() : m_name(""), m_size(0), m_data(NULL), m_allocator() {};
  AoS(Allocator& allocator) : m_name(""), m_size(0), m_data(NULL), m_allocator(allocator) {};

//  AoS(size_t size) : AoS() { 
//    m_size = size;
//    m_data = reinterpret_cast<T*>(m_allocator.allocate(size * sizeof(T)));
//  }
  AoS(Allocator& allocator, size_t size) : AoS(allocator) { 
    m_size = size; 
    m_data = reinterpret_cast<T*>(m_allocator.allocate(size * sizeof(T)));
  }

  ~AoS() { m_allocator.destroy(reinterpret_cast<char*>(m_data)); };

  T* data() const { return m_data; };
  size_t size() const { return m_size; };
};

template<class Allocator, class...  Args>
class SoA {
  std::tuple<Args...> m_data;
  size_t m_size;
  Allocator m_allocator;
  // TODO
  // if first arg has allocate()
  //   declare first arg as allocator
  //   pass the rest of args down 
  // else 
  //   declare allocator as std:: 
  //   pass all args down

  /* -- Allocators -- */
  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  init(std::tuple<Tp...>& t) {}

  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type
  init(std::tuple<Tp...>& t) {
    typedef decltype(std::get<I>(t)) t0; // reference to tuple contained object e.x float &*
    typedef typename std::remove_reference<t0>::type t1; // float *
    typedef typename std::remove_pointer<t1>::type t2; // float
    std::get<I>(t) = reinterpret_cast<t2*>(m_allocator.allocate(m_size * sizeof(t2)));
    init<I + 1, Tp...>(t);
  }

  /* -- Destructors -- */
  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I == sizeof...(Tp), void>::type
  kill(std::tuple<Tp...>& t) {}

  template<std::size_t I = 0, typename... Tp>
  inline typename std::enable_if<I < sizeof...(Tp), void>::type
  kill(std::tuple<Tp...>& t) {
    m_allocator.destroy(reinterpret_cast<char*>(std::get<I>(t)));
    kill<I + 1, Tp...>(t);
  }


  public:
  SoA(size_t size) : m_size{size}, m_data{}, m_allocator() {
    char* tttt = m_allocator.allocate(1);
    init(m_data);
  }; 

  SoA(Allocator& allocator, size_t size) : m_size{size}, m_data{}, m_allocator{allocator} {
    init(m_data);
  }; 

  ~SoA() {
    kill(m_data);
  }; 

  template<int I>
  auto data() {
    return std::get<I>(m_data);
  }


};

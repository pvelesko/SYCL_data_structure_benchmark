#include <iostream>

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

template<class T, class Allocator = std::allocator<T>>
class AoS {
  Allocator m_allocator;
  std::string m_name;
  size_t m_size;
  T* m_data;
  public:
  AoS() : m_name(""), m_size(0), m_data(NULL) { m_allocator = Allocator(); };
  AoS(Allocator allocator) : AoS() { m_allocator = allocator; };
  AoS(size_t size) : AoS() { 
    m_size = size;
    m_data = static_cast<T*>(m_allocator.allocate(size));
  }
  AoS(Allocator allocator, size_t size) : AoS(allocator) { 
    m_size = size; 
    m_data = static_cast<T*>(m_allocator.allocate(size));
  }

  T* data() const { return m_data; };
  size_t size() const { return m_size; };

};

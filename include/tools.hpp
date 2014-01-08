#ifndef TOOLS_HPP_
#define TOOLS_HPP_

// STL stuff
#include <complex>
#include <exception>
#include <fstream>
#include <iostream>

#include <assert.h>

namespace hifde3d {

typedef std::complex<double> cpx;

inline int pow2(int l) { assert(l >= 0); return (1 << l); }

template <class T, class S>
std::istream& operator>>(std::istream& is, std::pair<T,S>& a) {
  is >> a.first;
  is >> a.second;
  return is;
}
template <class T, class S>
std::ostream& operator<<(std::ostream& os, const std::pair<T,S>& a) {
  os << a.first << " " << a.second;
  return os;
}

}

#endif  // _COMMON_HPP_

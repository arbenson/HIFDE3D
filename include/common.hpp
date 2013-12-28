#ifndef _COMMON_HPP_
#define _COMMON_HPP_

// STL stuff
#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <complex>
#include <deque>
#include <exception>
#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

// external libraries
#include "blas.h"
#include "fftw3.h"
#include "mpi.h"

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

#ifndef RELEASE
void PushCallStack( std::string s );
void PopCallStack();
void DumpCallStack( std::ostream& os=std::cerr );

class CallStackEntry {
public:
    CallStackEntry( std::string s ) { 
        if( !std::uncaught_exception() )
            PushCallStack(s); 
    }
    ~CallStackEntry() { 
        if( !std::uncaught_exception() ) 
            PopCallStack(); 
    }
};
#endif  // ifndef RELEASE

#endif  // _COMMON_HPP_

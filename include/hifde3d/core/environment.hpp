/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (HIFDE3D) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef HIFDE3D_ENVIRONMENT_HPP
#define HIFDE3D_ENVIRONMENT_HPP 1

#include "mpi.h"
#include <algorithm>
#include <climits>
#include <complex>
#include <cstring>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <utility>
#include <vector>

#include "hifde3d/core/mpi.hpp"
#include "hifde3d/core/choice.hpp"
#include "hifde3d/core/mpi_choice.hpp"

namespace hifde3d {

bool Initialized();
void Initialize( int& argc, char**& argv );
void Finalize();
MpiArgs& GetArgs();

template<typename T>
inline T
Input( std::string name, std::string desc )
{ return GetArgs().Input<T>( name, desc ); }

template<typename T>
inline T
Input( std::string name, std::string desc, T defaultVal )
{ return GetArgs().Input( name, desc, defaultVal ); }

inline void
ProcessInput()
{ GetArgs().Process(); }

inline void
PrintInputReport()
{ GetArgs().PrintReport(); }

// These should typically only be used when not in RELEASE mode
#ifndef RELEASE
void PushCallStack( const std::string s );
void PopCallStack();
void DumpCallStack();

class CallStackEntry
{
public:
    CallStackEntry( std::string s )
    {
        if( !std::uncaught_exception() )
            PushCallStack(s);
    }
    ~CallStackEntry()
    {
        if( !std::uncaught_exception() )
            PopCallStack();
    }
};
#endif

int Oversample();
void SetOversample( int oversample );

template<typename Real>
Real CompressionTolerance();
template<>
float CompressionTolerance<float>();
template<>
double CompressionTolerance<double>();
template<typename Real>
void SetCompressionTolerance( Real relTol );
template<>
void SetCompressionTolerance<float>( float relTol );
template<>
void SetCompressionTolerance<double>( double relTol );

template<typename Real>
Real MidcomputeTolerance();
template<>
float MidcomputeTolerance<float>();
template<>
double MidcomputeTolerance<double>();
template<typename Real>
void SetMidcomputeTolerance( Real tolerance );
template<>
void SetMidcomputeTolerance<float>( float tolerance );
template<>
void SetMidcomputeTolerance<double>( double tolerance );

#ifdef MEMORY_INFO
void ResetMemoryCount( int key = -1 );
void NewMemoryCount( int key );
void EraseMemoryCount( int key );
void AddToMemoryCount( double size );
double MemoryUsage( int key = -1 );
double PeakMemoryUsage( int key = -1 );
#endif

void PrintGlobal
( double num, const std::string tag = "", std::ostream& os = std::cout );

typedef unsigned char byte;

template<typename T>
inline void
MemCopy( T* dest, const T* source, std::size_t numEntries )
{
    // This can be optimized/generalized later
    std::memcpy( dest, source, numEntries*sizeof(T) );
}

template<typename T>
inline void
MemZero( T* buffer, std::size_t numEntries )
{
    // This can be optimized/generalized later
    std::memset( buffer, 0, numEntries*sizeof(T) );
}

template<typename Real>
inline Real
Abs( Real alpha )
{ return std::abs(alpha); }

template<typename Real>
inline Real
Abs( std::complex<Real> alpha )
{ return std::abs(alpha); }

template<typename Real>
inline Real
Conj( Real alpha )
{ return alpha; }

template<typename Real>
inline std::complex<Real>
Conj( std::complex<Real> alpha )
{ return std::conj( alpha ); }

// For reading and writing to a buffer
template<typename T>
inline void Write( byte*& head, const T& t )
{
    *((T*)head) = t;
    head += sizeof(T);
}

template<typename T>
inline void Write( byte** head, const T& t )
{
    *((T*)*head) = t;
    *head += sizeof(T);
}

template<typename T>
inline void Write( byte*& head, const T* buffer, int n )
{
    std::memcpy( head, buffer, n*sizeof(T) );
    head += n*sizeof(T);
}

template<typename T>
inline void Write( byte** head, const T* buffer, int n )
{
    std::memcpy( *head, buffer, n*sizeof(T) );
    *head += n*sizeof(T);
}

template<typename T>
inline T Read( const byte*& head )
{
    T retval = *((const T*)head);
    head += sizeof(T);
    return retval;
}

template<typename T>
inline T Read( const byte** head )
{
    T retval = *((const T*)*head);
    *head += sizeof(T);
    return retval;
}

template<typename T>
inline void Read( T* writeHead, const byte*& readHead, int n )
{
    std::memcpy( writeHead, readHead, n*sizeof(T) );
    readHead += n*sizeof(T);
}

template<typename T>
inline void Read( T* writeHead, const byte** readHead, int n )
{
    std::memcpy( writeHead, *readHead, n*sizeof(T) );
    *readHead += n*sizeof(T);
}

// For extracting the underlying real datatype,
// e.g., typename Base<Scalar>::type a = 3.0;
template<typename Real>
struct Base
{ typedef Real type; };

template<typename Real>
struct Base<std::complex<Real> >
{ typedef Real type; };

#define BASE(F) typename hifde3d::Base<F>::type

// Create a wrappers around real and std::complex<real> types so that they
// can be conveniently printed in a more Matlab-compatible format.
//
// All printing of scalars should now be performed in the fashion:
//     std::cout << WrapScalar(alpha);
// where 'alpha' can be real or complex.

template<typename Real>
class ScalarWrapper
{
    const Real value_;
public:
    ScalarWrapper( const Real alpha ) : value_(alpha) { }

    friend std::ostream& operator<<
    ( std::ostream& out, const ScalarWrapper<Real> alpha )
    {
        out << alpha.value_;
        return out;
    }
};

template<typename Real>
class ScalarWrapper<std::complex<Real> >
{
    const std::complex<Real> value_;
public:
    ScalarWrapper( const std::complex<Real> alpha ) : value_(alpha) { }

    friend std::ostream& operator<<
    ( std::ostream& os, const ScalarWrapper<std::complex<Real> > alpha )
    {
        os << std::real(alpha.value_) << "+" << std::imag(alpha.value_) << "i";
        return os;
    }
};

template<typename Real>
inline const ScalarWrapper<Real>
WrapScalar( const Real alpha )
{ return ScalarWrapper<Real>( alpha ); }

template<typename Real>
inline const ScalarWrapper<std::complex<Real> >
WrapScalar( const std::complex<Real> alpha )
{ return ScalarWrapper<std::complex<Real> >( alpha ); }

inline unsigned Log2( unsigned N )
{
#ifndef RELEASE
    CallStackEntry entry("Log2");
    if( N == 0 )
        throw std::logic_error("Cannot take integer log2 of 0");
#endif
    int result = 0;
    if( N >= (1<<16)) { N >>= 16; result += 16; }
    if( N >= (1<< 8)) { N >>=  8; result +=  8; }
    if( N >= (1<< 4)) { N >>=  4; result +=  4; }
    if( N >= (1<< 2)) { N >>=  2; result +=  2; }
    if( N >= (1<< 1)) { N >>=  1; result +=  1; }
    return result;
}

inline void AddToMap
( std::map<int,int>& map, int key, int value )
{
    if( value == 0 )
        return;

    std::map<int,int>::iterator it;
    it = map.find( key );
    if( it == map.end() )
        map[key] = value;
    else
        it->second += value;
}

} // namespace hifde3d

#endif // ifndef HIFDE3D_ENVIRONMENT_HPP

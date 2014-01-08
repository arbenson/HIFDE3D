/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (HIFDE3D) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef HIFDE3D_LAPACK_HPP
#define HIFDE3D_LAPACK_HPP 1

#include "blas.hpp"
#include <complex>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>

#if defined(LAPACK_POST)
#define LAPACK(name) name ## _
#else
#define LAPACK(name) name
#endif

#define BLOCKSIZE 32

extern "C" {

typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;

float LAPACK(slamch)( const char* cmach );
double LAPACK(dlamch)( const char* cmach );

float LAPACK(slapy2)
( const float* alpha, const float* beta );
double LAPACK(dlapy2)
( const double* alpha, const double* beta );

float LAPACK(slapy3)
( const float* alpha, const float* beta, const float* gamma );
double LAPACK(dlapy3)
( const double* alpha, const double* beta, const double* gamma );

void LAPACK(sgeqrf)
( const int* m, const int* n,
  float* A, const int* lda,
  float* tau,
  float* work, const int* lwork,
  int* info );

void LAPACK(dgeqrf)
( const int* m, const int* n,
  double* A, const int* lda,
  double* tau,
  double* work, const int* lwork,
  int* info );

void LAPACK(cgeqrf)
( const int* m, const int* n,
  scomplex* A, const int* lda,
  scomplex* tau,
  scomplex* work, const int* lwork,
  int* info );

void LAPACK(zgeqrf)
( const int* m, const int* n,
  dcomplex* A, const int* lda,
  dcomplex* tau,
  dcomplex* work, const int* lwork,
  int* info );

void LAPACK(sgeqp3)
( const int* m, const int* n,
  float* A, const int* lda,
  int* jpvt,
  float* tau,
  float* work, const int* lwork,
  int* info );

void LAPACK(dgeqp3)
( const int* m, const int* n,
  double* A, const int* lda,
  int* jpvt,
  double* tau,
  double* work, const int* lwork,
  int* info );

void LAPACK(cgeqp3)
( const int* m, const int* n,
  scomplex* A, const int* lda,
  int* jpvt,
  scomplex* tau,
  scomplex* work, const int* lwork,
  float* rwork,
  int* info );

void LAPACK(zgeqp3)
( const int* m, const int* n,
  dcomplex* A, const int* lda,
  int* jpvt,
  dcomplex* tau,
  dcomplex* work, const int* lwork,
  double* rwork,
  int* info );

void LAPACK(sormqr)
( const char* side, const char* trans,
  const int* m, const int* n, const int* k,
  const float* A, const int* lda,
  const float* tau,
  float* C, const int* ldc,
  float* work, const int* lwork,
  int* info );

void LAPACK(dormqr)
( const char* side, const char* trans,
  const int* m, const int* n, const int* k,
  const double* A, const int* lda,
  const double* tau,
  double* C, const int* ldc,
  double* work, const int* lwork,
  int* info );

void LAPACK(cunmqr)
( const char* side, const char* trans,
  const int* m, const int* n, const int* k,
  const scomplex* A, const int* lda,
  const scomplex* tau,
  scomplex* C, const int* ldc,
  scomplex* work, const int* lwork,
  int* info );

void LAPACK(zunmqr)
( const char* side, const char* trans,
  const int* m, const int* n, const int* k,
  const dcomplex* A, const int* lda,
  const dcomplex* tau,
  dcomplex* C, const int* ldc,
  dcomplex* work, const int* lwork,
  int* info );

void LAPACK(sorgqr)
( const int* m, const int* n, const int* k,
        float* A, const int* lda,
  const float* tau,
        float* work, const int* lwork,
  int* info );

void LAPACK(dorgqr)
( const int* m, const int* n, const int* k,
        double* A, const int* lda,
  const double* tau,
        double* work, const int* lwork,
  int* info );

void LAPACK(cungqr)
( const int* m, const int* n, const int* k,
        scomplex* A, const int* lda,
  const scomplex* tau,
        scomplex* work, const int* lwork,
  int* info );

void LAPACK(zungqr)
( const int* m, const int* n, const int* k,
        dcomplex* A, const int* lda,
  const dcomplex* tau,
        dcomplex* work, const int* lwork,
  int* info );

void LAPACK(sgesvd)
( const char* jobu, const char* jobvh,
  const int* m, const int* n,
  float* A, const int* lda,
  float* s,
  float* U, const int* ldu,
  float* VH, const int* ldvh,
  float* work, const int* lwork,
  int* info );

void LAPACK(dgesvd)
( const char* jobu, const char* jobvh,
  const int* m, const int* n,
  double* A, const int* lda,
  double* s,
  double* U, const int* ldu,
  double* VH, const int* ldvh,
  double* work, const int* lwork,
  int* info );

void LAPACK(cgesvd)
( const char* jobu, const char* jobvh,
  const int* m, const int* n,
  scomplex* A, const int* lda,
  float* s,
  scomplex* U, const int* ldu,
  scomplex* VH, const int* ldvh,
  scomplex* work, const int* lwork,
  float* rwork,
  int* info );

void LAPACK(zgesvd)
( const char* jobu, const char* jobvh,
  const int* m, const int* n,
  dcomplex* A, const int* lda,
  double* s,
  dcomplex* U, const int* ldu,
  dcomplex* VH, const int* ldvh,
  dcomplex* work, const int* lwork,
  double* rwork,
  int* info );

float LAPACK(slamch)( const char* cmach );
double LAPACK(dlamch)( const char* cmach );

void LAPACK(sgetrf)
( const int* m, const int* n,
  float* A, const int* lda,
  int* ipiv,
  int* info );

void LAPACK(dgetrf)
( const int* m, const int* n,
  double* A, const int* lda,
  int* ipiv,
  int* info );

void LAPACK(cgetrf)
( const int* m, const int* n,
  scomplex* A, const int* lda,
  int* ipiv,
  int* info );

void LAPACK(zgetrf)
( const int* m, const int* n,
  dcomplex* A, const int* lda,
  int* ipiv,
  int* info );

void LAPACK(sgetri)
( const int* n,
  float* A, const int* lda,
  const int* ipiv,
  float* work, const int* lwork,
  int* info );

void LAPACK(dgetri)
( const int* n,
  double* A, const int* lda,
  const int* ipiv,
  double* work, const int* lwork,
  int* info );

void LAPACK(cgetri)
( const int* n,
  scomplex* A, const int* lda,
  const int* ipiv,
  scomplex* work, const int* lwork,
  int* info );

void LAPACK(zgetri)
( const int* n,
  dcomplex* A, const int* lda,
  const int* ipiv,
  dcomplex* work, const int* lwork,
  int* info );

void LAPACK(ssytrf)
( const char* uplo, const int* n,
  float* A, const int* lda,
  int* ipiv,
  float* work, const int* lwork,
  int* info );

void LAPACK(dsytrf)
( const char* uplo, const int* n,
  double* A, const int* lda,
  int* ipiv,
  double* work, const int* lwork,
  int* info );

void LAPACK(csytrf)
( const char* uplo, const int* n,
  scomplex* A, const int* lda,
  int* ipiv,
  scomplex* work, const int* lwork,
  int* info );

void LAPACK(zsytrf)
( const char* uplo, const int* n,
  dcomplex* A, const int* lda,
  int* ipiv,
  dcomplex* work, const int* lwork,
  int* info );

void LAPACK(ssytri)
( const char* uplo, const int* n,
  float* A, const int* lda,
  int* ipiv,
  float* work,
  int* info );

void LAPACK(dsytri)
( const char* uplo, const int* n,
  double* A, const int* lda,
  int* ipiv,
  double* work,
  int* info );

void LAPACK(csytri)
( const char* uplo, const int* n,
  scomplex* A, const int* lda,
  int* ipiv,
  scomplex* work,
  int* info );

void LAPACK(zsytri)
( const char* uplo, const int* n,
  dcomplex* A, const int* lda,
  int* ipiv,
  dcomplex* work,
  int* info );

void LAPACK(ssyevd)
( const char* jobz, const char* uplo,
  const int* n,
  float* A, const int* lda,
  float* w,
  float* work, const int* lwork,
  int* iwork, const int* liwork,
  int* info );

void LAPACK(dsyevd)
( const char* jobz, const char* uplo,
  const int* n,
  double* A, const int* lda,
  double* w,
  double* work, const int* lwork,
  int* iwork, const int* liwork,
  int* info );

void LAPACK(cheevd)
( const char* jobz, const char* uplo,
  const int* n,
  scomplex* A, const int* lda,
  float* w,
  scomplex* work, const int* lwork,
  float* rwork, const int* lrwork,
  int* iwork, const int* liwork,
  int* info );

void LAPACK(zheevd)
( const char* jobz, const char* uplo,
  const int* n,
  dcomplex* A, const int* lda,
  double* w,
  dcomplex* work, const int* lwork,
  double* rwork, const int* lrwork,
  int* iwork, const int* liwork,
  int* info );

} // extern "C"

namespace hifde3d {
namespace lapack {

//----------------------------------------------------------------------------//
// Machine constants                                                          //
//----------------------------------------------------------------------------//

template<typename Real> Real MachineEpsilon();

template<>
inline float MachineEpsilon<float>()
{
    const char cmach = 'E';
    return LAPACK(slamch)( &cmach );
}

template<>
inline double MachineEpsilon<double>()
{
    const char cmach = 'E';
    return LAPACK(dlamch)( &cmach );
}

template<typename Real> Real MachineSafeMin();

template<>
inline float MachineSafeMin()
{
    const char cmach = 'S';
    return LAPACK(slamch)( &cmach );
}

template<>
inline double MachineSafeMin()
{
    const char cmach = 'S';
    return LAPACK(dlamch)( &cmach );
}

//----------------------------------------------------------------------------//
// Safe Norms (avoid under/over-flow)                                         //
//----------------------------------------------------------------------------//

inline float SafeNorm( float alpha, float beta )
{ return LAPACK(slapy2)( &alpha, &beta ); }

inline double SafeNorm( double alpha, double beta )
{ return LAPACK(dlapy2)( &alpha, &beta ); }

inline float SafeNorm( float alpha, float beta, float gamma )
{ return LAPACK(slapy3)( &alpha, &beta, &gamma ); }

inline double SafeNorm( double alpha, double beta, double gamma )
{ return LAPACK(dlapy3)( &alpha, &beta, &gamma ); }

//----------------------------------------------------------------------------//
// Unpivoted QR                                                               //
//----------------------------------------------------------------------------//

inline int QRWorkSize( int n )
{
    // Minimum workspace for using blocksize BLOCKSIZE.
    return std::max(1,n*BLOCKSIZE);
}

inline void QR
( int m, int n,
  float* A, int lda,
  float* tau,
  float* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::QR");
    if( lda < std::max(1,m) )
        throw std::logic_error("lda was too small");
    if( lwork < std::max(1,n) )
        throw std::logic_error("lwork too small");
#endif
    int info;
    LAPACK(sgeqrf)( &m, &n, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, sgeqrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void QR
( int m, int n,
  double* A, int lda,
  double* tau,
  double* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::QR");
    if( lda < std::max(1,m) )
        throw std::logic_error("lda was too small");
    if( lwork < std::max(1,n) )
        throw std::logic_error("lwork too small");
#endif
    int info;
    LAPACK(dgeqrf)( &m, &n, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, dgeqrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void QR
( int m, int n,
  std::complex<float>* A, int lda,
  std::complex<float>* tau,
  std::complex<float>* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::QR");
    if( lda < std::max(1,m) )
        throw std::logic_error("lda was too small");
    if( lwork < std::max(1,n) )
        throw std::logic_error("lwork too small");
#endif
    int info;
    LAPACK(cgeqrf)( &m, &n, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, cgeqrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void QR
( int m, int n,
  std::complex<double>* A, int lda,
  std::complex<double>* tau,
  std::complex<double>* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::QR");
    if( lda < std::max(1,m) )
        throw std::logic_error("lda was too small");
    if( lwork < std::max(1,n) )
        throw std::logic_error("lwork too small");
#endif
    int info;
    LAPACK(zgeqrf)( &m, &n, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, zgeqrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// Pivoted QR                                                                 //
//----------------------------------------------------------------------------//

inline int PivotedQRWorkSize( int n )
{
    // Return the amount of space needed for a blocksize of BLOCKSIZE.
    return 2*n + (n+1)*BLOCKSIZE;
}

inline int PivotedQRRealWorkSize( int n )
{
    return 2*n;
}

inline void PivotedQR
( int m, int n,
  float* A, int lda,
  int* jpvt,
  float* tau,
  float* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::PivotedQR");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(sgeqp3)( &m, &n, A, &lda, jpvt, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, sgeqp3, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void PivotedQR
( int m, int n,
  double* A, int lda,
  int* jpvt,
  double* tau,
  double* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::PivotedQR");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(dgeqp3)( &m, &n, A, &lda, jpvt, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, dgeqp3, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void PivotedQR
( int m, int n,
  std::complex<float>* A, int lda,
  int* jpvt,
  std::complex<float>* tau,
  std::complex<float>* work, int lwork,
  float* rwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::PivotedQR");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(cgeqp3)( &m, &n, A, &lda, jpvt, tau, work, &lwork, rwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, cgeqp3, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void PivotedQR
( int m, int n,
  std::complex<double>* A, int lda,
  int* jpvt,
  std::complex<double>* tau,
  std::complex<double>* work, int lwork,
  double* rwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::PivotedQR");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(zgeqp3)( &m, &n, A, &lda, jpvt, tau, work, &lwork, rwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "QR factorization, zgeqp3, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// Apply Q from a QR factorization                                            //
//----------------------------------------------------------------------------//

inline int ApplyQWorkSize( char side, int m, int n )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::ApplyQWorkSize");
#endif
    int worksize;
    if( side == 'L' )
        worksize = std::max(1,n*BLOCKSIZE);
    else if( side == 'R' )
        worksize = std::max(1,m*BLOCKSIZE);
#ifndef RELEASE
    else
        throw std::logic_error("Invalid side for ApplyQ worksize query.");
#endif
    return worksize;
}

inline void ApplyQ
( char side, char trans, int m, int n, int k,
  const float* A, int lda,
  const float* tau,
        float* C, int ldc,
        float* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::ApplyQ");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldc == 0 )
        throw std::logic_error("ldc was 0");
#endif
    // Convert the more general complex option to the real case
    if( trans == 'C' || trans == 'c' )
        trans = 'T';

    int info;
    LAPACK(sormqr)
    ( &side, &trans, &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, sormqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void ApplyQ
( char side, char trans, int m, int n, int k,
  const double* A, int lda,
  const double* tau,
        double* C, int ldc,
        double* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::ApplyQ");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldc == 0 )
        throw std::logic_error("ldc was 0");
#endif
    // Convert the more general complex option to the real case
    if( trans == 'C' || trans == 'c' )
        trans = 'T';

    int info;
    LAPACK(dormqr)
    ( &side, &trans, &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, dormqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void ApplyQ
( char side, char trans, int m, int n, int k,
  const std::complex<float>* A, int lda,
  const std::complex<float>* tau,
        std::complex<float>* C, int ldc,
        std::complex<float>* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::ApplyQ");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldc == 0 )
        throw std::logic_error("ldc was 0");
#endif
    int info;
    LAPACK(cunmqr)
    ( &side, &trans, &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, cunmqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void ApplyQ
( char side, char trans, int m, int n, int k,
  const std::complex<double>* A, int lda,
  const std::complex<double>* tau,
        std::complex<double>* C, int ldc,
        std::complex<double>* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::ApplyQ");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldc == 0 )
        throw std::logic_error("ldc was 0");
#endif
    int info;
    LAPACK(zunmqr)
    ( &side, &trans, &m, &n, &k, A, &lda, tau, C, &ldc, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, zunmqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// Form Q from a QR factorization                                             //
//----------------------------------------------------------------------------//

inline int FormQWorkSize( int n )
{
    // Minimum workspace for using blocksize BLOCKSIZE.
    return std::max(1,n*BLOCKSIZE);
}

inline void FormQ
( int m, int n, int k,
        float* A, int lda,
  const float* tau,
        float* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::FormQ");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(sorgqr)( &m, &n, &k, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, sorgqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void FormQ
( int m, int n, int k,
        double* A, int lda,
  const double* tau,
        double* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::FormQ");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(dorgqr)( &m, &n, &k, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, dorgqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void FormQ
( int m, int n, int k,
        std::complex<float>* A, int lda,
  const std::complex<float>* tau,
        std::complex<float>* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::FormQ");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(cungqr)( &m, &n, &k, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, cungqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void FormQ
( int m, int n, int k,
        std::complex<double>* A, int lda,
  const std::complex<double>* tau,
        std::complex<double>* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::FormQ");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(zungqr)( &m, &n, &k, A, &lda, tau, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "Q application, zungqr, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// SVD                                                                        //
//----------------------------------------------------------------------------//

inline int SVDWorkSize( int m, int n )
{
    // Return 5 times the minimum recommendation
    return 5*std::max( 1, 2*std::min(m,n)+std::max(m,n) );
}

inline int SVDRealWorkSize( int m, int n )
{
    return 5*5*std::min( m, n );
}

inline void SVD
( char jobu, char jobvh, int m, int n,
  float* A, int lda,
  float* s,
  float* U, int ldu,
  float* VH, int ldvh,
  float* work, int lwork, __attribute__((unused)) float* rwork=0 )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::SVD");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldu == 0 )
        throw std::logic_error("ldu was 0");
    if( ldvh == 0 )
        throw std::logic_error("ldvh was 0");
    if( jobu == 'A' && ldu < m )
        throw std::logic_error("ldu too small for jobu='A'");
    else if( jobu == 'S' && ldu < std::min(m,n) )
        throw std::logic_error("ldu too small for jobu='S'");
    if( jobvh == 'A' && ldvh < n )
        throw std::logic_error("ldvh too small for jobvh='A'");
    else if( jobvh == 'S' && ldvh < std::min(m,n) )
        throw std::logic_error("ldvh too small for jobvh='S'");
#endif
    int info;
    LAPACK(sgesvd)
    ( &jobu, &jobvh, &m, &n, A, &lda, s, U, &ldu, VH, &ldvh, work, &lwork,
      &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "SVD, sgesvd, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void SVD
( char jobu, char jobvh, int m, int n,
  double* A, int lda,
  double* s,
  double* U, int ldu,
  double* VH, int ldvh,
  double* work, int lwork, __attribute__((unused)) double* rwork=0 )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::SVD");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldu == 0 )
        throw std::logic_error("ldu was 0");
    if( ldvh == 0 )
        throw std::logic_error("ldvh was 0");
    if( jobu == 'A' && ldu < m )
        throw std::logic_error("ldu too small for jobu='A'");
    else if( jobu == 'S' && ldu < std::min(m,n) )
        throw std::logic_error("ldu too small for jobu='S'");
    if( jobvh == 'A' && ldvh < n )
        throw std::logic_error("ldvh too small for jobvh='A'");
    else if( jobvh == 'S' && ldvh < std::min(m,n) )
        throw std::logic_error("ldvh too small for jobvh='S'");
#endif
    int info;
    LAPACK(dgesvd)
    ( &jobu, &jobvh, &m, &n, A, &lda, s, U, &ldu, VH, &ldvh, work, &lwork,
      &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "SVD, dgesvd, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void SVD
( char jobu, char jobvh, int m, int n,
  std::complex<float>* A, int lda,
  float* s,
  std::complex<float>* U, int ldu,
  std::complex<float>* VH, int ldvh,
  std::complex<float>* work, int lwork,
  float* rwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::SVD");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldu == 0 )
        throw std::logic_error("ldu was 0");
    if( ldvh == 0 )
        throw std::logic_error("ldvh was 0");
    if( jobu == 'A' && ldu < m )
        throw std::logic_error("ldu too small for jobu='A'");
    else if( jobu == 'S' && ldu < std::min(m,n) )
        throw std::logic_error("ldu too small for jobu='S'");
    if( jobvh == 'A' && ldvh < n )
        throw std::logic_error("ldvh too small for jobvh='A'");
    else if( jobvh == 'S' && ldvh < std::min(m,n) )
        throw std::logic_error("ldvh too small for jobvh='S'");
#endif
    int info;
    LAPACK(cgesvd)
    ( &jobu, &jobvh, &m, &n, A, &lda, s, U, &ldu, VH, &ldvh, work, &lwork,
      rwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "SVD, cgesvd, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void SVD
( char jobu, char jobvh, int m, int n,
  std::complex<double>* A, int lda,
  double* s,
  std::complex<double>* U, int ldu,
  std::complex<double>* VH, int ldvh,
  std::complex<double>* work, int lwork,
  double* rwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::SVD");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldu == 0 )
        throw std::logic_error("ldu was 0");
    if( ldvh == 0 )
        throw std::logic_error("ldvh was 0");
    if( jobu == 'A' && ldu < m )
        throw std::logic_error("ldu too small for jobu='A'");
    else if( jobu == 'S' && ldu < std::min(m,n) )
        throw std::logic_error("ldu too small for jobu='S'");
    if( jobvh == 'A' && ldvh < n )
        throw std::logic_error("ldvh too small for jobvh='A'");
    else if( jobvh == 'S' && ldvh < std::min(m,n) )
        throw std::logic_error("ldvh too small for jobvh='S'");
#endif
    int info;
    LAPACK(zgesvd)
    ( &jobu, &jobvh, &m, &n, A, &lda, s, U, &ldu, VH, &ldvh, work, &lwork,
      rwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "SVD, zgesvd, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void SVD
( char jobu, char jobvh, int m, int n,
  float* A, int lda,
  float* s,
  float* U, int ldu,
  float* VH, int ldvh )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::SVD");
#endif
    const int lwork = SVDWorkSize( m, n );
    std::vector<float> work(lwork);
    SVD( jobu, jobvh, m, n, A, lda, s, U, ldu, VH, ldvh, &work[0], lwork );
}

inline void SVD
( char jobu, char jobvh, int m, int n,
  double* A, int lda,
  double* s,
  double* U, int ldu,
  double* VH, int ldvh )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::SVD");
#endif
    const int lwork = SVDWorkSize( m, n );
    std::vector<double> work(lwork);
    SVD
    ( jobu, jobvh, m, n, A, lda, s, U, ldu, VH, ldvh,
      &work[0], lwork );
}

inline void SVD
( char jobu, char jobvh, int m, int n,
  std::complex<float>* A, int lda,
  float* s,
  std::complex<float>* U, int ldu,
  std::complex<float>* VH, int ldvh )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::SVD");
#endif
    const int lwork = SVDWorkSize( m, n );
    const int lrwork = SVDRealWorkSize( m, n );
    std::vector<std::complex<float> > work(lwork);
    std::vector<float> rwork(lrwork);
    SVD
    ( jobu, jobvh, m, n, A, lda, s, U, ldu, VH, ldvh,
      &work[0], lwork, &rwork[0] );
}

inline void SVD
( char jobu, char jobvh, int m, int n,
  std::complex<double>* A, int lda,
  double* s,
  std::complex<double>* U, int ldu,
  std::complex<double>* VH, int ldvh )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::SVD");
#endif
    const int lwork = SVDWorkSize( m, n );
    const int lrwork = SVDRealWorkSize( m, n );
    std::vector<std::complex<double> > work(lwork);
    std::vector<double> rwork(lrwork);
    SVD
    ( jobu, jobvh, m, n, A, lda, s, U, ldu, VH, ldvh,
      &work[0], lwork, &rwork[0] );
}
//----------------------------------------------------------------------------//
// Adjoint Pseudo-inverse (using an SVD)                                      //
//----------------------------------------------------------------------------//

inline void AdjointPseudoInverse
( int m, int n,
  float* A, int lda,
  float* s,
  float* U, int ldu,
  float* VH, int ldvh,
  float* work, int lwork,
  __attribute__((unused)) float* realWork, float epsilon )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::AdjointPseudoInverse");
#endif
    const int minDim = std::min( m, n );
    const int maxDim = std::max( m, n );
    if( minDim == 0 )
        return;

    // Clearly we could perform the cutoff within this routine...
    lapack::SVD( 'S', 'S', m, n, A, lda, s, U, ldu, VH, ldvh, work, lwork );

    // Scale the columns of U using thresholded inversion of the singular values
    const float tolerance = s[0]*maxDim*epsilon;
    int cutoff;
    for( cutoff=0; cutoff<minDim; ++cutoff )
    {
        if( s[cutoff] > tolerance )
            for( int i=0; i<m; ++i )
                U[i+cutoff*ldu] /= s[cutoff];
        else
            break;
    }

    // Form A := [U pinv(Sigma)] V^H
    blas::Gemm( 'N', 'N', m, n, cutoff, 1, U, ldu, VH, ldvh, 0, A, lda );
}

inline void AdjointPseudoInverse
( int m, int n,
  double* A, int lda,
  double* s,
  double* U, int ldu,
  double* VH, int ldvh,
  double* work, int lwork,
  __attribute__((unused)) double* realWork, double epsilon )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::AdjointPseudoInverse");
#endif
    const int minDim = std::min( m, n );
    const int maxDim = std::max( m, n );
    if( minDim == 0 )
        return;

    // Clearly we could perform the cutoff within this routine...
    lapack::SVD( 'S', 'S', m, n, A, lda, s, U, ldu, VH, ldvh, work, lwork );

    // Scale the columns of U using thresholded inversion of the singular values
    const double tolerance = s[0]*maxDim*epsilon;
    int cutoff;
    for( cutoff=0; cutoff<minDim; ++cutoff )
    {
        if( s[cutoff] > tolerance )
            for( int i=0; i<m; ++i )
                U[i+cutoff*ldu] /= s[cutoff];
        else
            break;
    }

    // Form A := [U pinv(Sigma)] V^H
    blas::Gemm( 'N', 'N', m, n, cutoff, 1, U, ldu, VH, ldvh, 0, A, lda );
}

inline void AdjointPseudoInverse
( int m, int n,
  std::complex<float>* A, int lda,
  float* s,
  std::complex<float>* U, int ldu,
  std::complex<float>* VH, int ldvh,
  std::complex<float>* work, int lwork,
  float* rwork, float epsilon )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::AdjointPseudoInverse");
#endif
    const int minDim = std::min( m, n );
    const int maxDim = std::max( m, n );
    if( minDim == 0 )
        return;

    // Clearly we could perform the cutoff within this routine...
    lapack::SVD
    ( 'S', 'S', m, n, A, lda, s, U, ldu, VH, ldvh, work, lwork, rwork );

    // Scale the columns of U using thresholded inversion of the singular values
    const float tolerance = s[0]*maxDim*epsilon;
    int cutoff;
    for( cutoff=0; cutoff<minDim; ++cutoff )
    {
        if( s[cutoff] > tolerance )
            for( int i=0; i<m; ++i )
                U[i+cutoff*ldu] /= s[cutoff];
        else
            break;
    }

    // Form A := [U pinv(Sigma)] V^H
    blas::Gemm( 'N', 'N', m, n, cutoff, 1, U, ldu, VH, ldvh, 0, A, lda );
}

inline void AdjointPseudoInverse
( int m, int n,
  std::complex<double>* A, int lda,
  double* s,
  std::complex<double>* U, int ldu,
  std::complex<double>* VH, int ldvh,
  std::complex<double>* work, int lwork,
  double* rwork, double epsilon )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::AdjointPseudoInverse");
#endif
    const int minDim = std::min( m, n );
    const int maxDim = std::max( m, n );
    if( minDim == 0 )
        return;

    // Clearly we could perform the cutoff within this routine...
    lapack::SVD
    ( 'S', 'S', m, n, A, lda, s, U, ldu, VH, ldvh, work, lwork, rwork );

    // Scale the columns of U using thresholded inversion of the singular values
    const double tolerance = s[0]*maxDim*epsilon;
    int cutoff;
    for( cutoff=0; cutoff<minDim; ++cutoff )
    {
        if( s[cutoff] > tolerance )
            for( int i=0; i<m; ++i )
                U[i+cutoff*ldu] /= s[cutoff];
        else
            break;
    }

    // Form A := [U pinv(Sigma)] V^H
    blas::Gemm( 'N', 'N', m, n, cutoff, 1, U, ldu, VH, ldvh, 0, A, lda );
}

inline void AdjointPseudoInverse
( int m, int n,
  float* A, int lda,
  float* s,
  float* U, int ldu,
  float* VH, int ldvh,
  float* work, int lwork,
  float* realWork=0 )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::AdjointPseudoInverse");
#endif
    const float epsilon = MachineEpsilon<float>();
    AdjointPseudoInverse
    ( m, n, A, lda, s, U, ldu, VH, ldvh, work, lwork, realWork, epsilon );
}

inline void AdjointPseudoInverse
( int m, int n,
  double* A, int lda,
  double* s,
  double* U, int ldu,
  double* VH, int ldvh,
  double* work, int lwork,
  double* realWork=0 )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::AdjointPseudoInverse");
#endif
    const double epsilon = MachineEpsilon<double>();
    AdjointPseudoInverse
    ( m, n, A, lda, s, U, ldu, VH, ldvh, work, lwork, realWork, epsilon );
}

inline void AdjointPseudoInverse
( int m, int n,
  std::complex<float>* A, int lda,
  float* s,
  std::complex<float>* U, int ldu,
  std::complex<float>* VH, int ldvh,
  std::complex<float>* work, int lwork,
  float* rwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::AdjointPseudoInverse");
#endif
    const float epsilon = MachineEpsilon<float>();
    AdjointPseudoInverse
    ( m, n, A, lda, s, U, ldu, VH, ldvh, work, lwork, rwork, epsilon );
}

inline void AdjointPseudoInverse
( int m, int n,
  std::complex<double>* A, int lda,
  double* s,
  std::complex<double>* U, int ldu,
  std::complex<double>* VH, int ldvh,
  std::complex<double>* work, int lwork,
  double* rwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::AdjointPseudoInverse");
#endif
    const double epsilon = MachineEpsilon<double>();
    AdjointPseudoInverse
    ( m, n, A, lda, s, U, ldu, VH, ldvh, work, lwork, rwork, epsilon );
}

inline void AdjointPseudoInverse
( int m, int n, float* A, int lda, float epsilon )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::AdjointPseudoInverse");
#endif
    const int minDim = std::min( m, n );
    if( minDim == 0 )
        return;

    const int lwork = SVDWorkSize( m, n );
    std::vector<float> s(minDim), U(m*minDim), VH(minDim*n),
                       work(lwork);
    AdjointPseudoInverse
    ( m, n, A, lda, &s[0], &U[0], m, &VH[0], minDim,
      &work[0], lwork, 0, epsilon );
}

inline void AdjointPseudoInverse
( int m, int n, double* A, int lda, double epsilon )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::AdjointPseudoInverse");
#endif
    const int minDim = std::min( m, n );
    if( minDim == 0 )
        return;

    const int lwork = SVDWorkSize( m, n );
    std::vector<double> s(minDim), U(m*minDim), VH(minDim*n),
                        work(lwork);
    AdjointPseudoInverse
    ( m, n, A, lda, &s[0], &U[0], m, &VH[0], minDim,
      &work[0], lwork, 0, epsilon );
}

inline void AdjointPseudoInverse
( int m, int n, std::complex<float>* A, int lda, float epsilon )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::AdjointPseudoInverse");
#endif
    const int minDim = std::min( m, n );
    if( minDim == 0 )
        return;

    const int lwork = SVDWorkSize( m, n );
    const int lrwork = SVDRealWorkSize( m, n );
    std::vector<float> s(minDim), rwork(lrwork);
    std::vector<std::complex<float> > U(m*minDim), VH(minDim*n), work(lwork);
    AdjointPseudoInverse
    ( m, n, A, lda, &s[0], &U[0], m, &VH[0], minDim,
      &work[0], lwork, &rwork[0], epsilon );
}

inline void AdjointPseudoInverse
( int m, int n, std::complex<double>* A, int lda, double epsilon )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::AdjointPseudoInverse");
#endif
    const int minDim = std::min( m, n );
    if( minDim == 0 )
        return;

    const int lwork = SVDWorkSize( m, n );
    const int lrwork = SVDRealWorkSize( m, n );
    std::vector<double> s(minDim), rwork(lrwork);
    std::vector<std::complex<double> > U(m*minDim), VH(minDim*n), work(lwork);
    AdjointPseudoInverse
    ( m, n, A, lda, &s[0], &U[0], m, &VH[0], minDim,
      &work[0], lwork, &rwork[0], epsilon );
}

//----------------------------------------------------------------------------//
// LU Factorization                                                           //
//----------------------------------------------------------------------------//

inline void LU( int m, int n, float* A, int lda, int* ipiv )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::LU");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(sgetrf)( &m, &n, A, &lda, ipiv, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LU, sgetrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void LU( int m, int n, double* A, int lda, int* ipiv )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::LU");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(dgetrf)( &m, &n, A, &lda, ipiv, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LU, dgetrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void LU( int m, int n, std::complex<float>* A, int lda, int* ipiv )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::LU");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(cgetrf)( &m, &n, A, &lda, ipiv, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LU, cgetrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void LU( int m, int n, std::complex<double>* A, int lda, int* ipiv )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::LU");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(zgetrf)( &m, &n, A, &lda, ipiv, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LU, zgetrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// Invert an LU factorization                                                 //
//----------------------------------------------------------------------------//

inline int InvertLUWorkSize( int n )
{
    // Minimum space for running with blocksize BLOCKSIZE.
    return std::max(1,n*BLOCKSIZE);
}

inline void InvertLU
( int n, float* A, int lda, int* ipiv, float* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::InvertLU");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(sgetri)( &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLU, sgetri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void InvertLU
( int n, double* A, int lda, int* ipiv, double* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::InvertLU");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(dgetri)( &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLU, dgetri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void InvertLU
( int n, std::complex<float>* A, int lda, int* ipiv,
  std::complex<float>* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::InvertLU");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(cgetri)( &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLU, cgetri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void InvertLU
( int n, std::complex<double>* A, int lda, int* ipiv,
  std::complex<double>* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::InvertLU");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(zgetri)( &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLU, zgetri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// LDL^T Factorization                                                        //
//----------------------------------------------------------------------------//

inline int LDLTWorkSize( int n )
{
    // Return the worksize for a blocksize of BLOCKSIZE.
    return (n+1)*BLOCKSIZE;
}

inline void LDLT
( char uplo, int n, float* A, int lda, int* ipiv,
  float* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::LDLT");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(ssytrf)( &uplo, &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LDL^T, ssytrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void LDLT
( char uplo, int n, double* A, int lda, int* ipiv,
  double* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::LDLT");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(dsytrf)( &uplo, &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LDL^T, dsytrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void LDLT
( char uplo, int n, std::complex<float>* A, int lda, int* ipiv,
  std::complex<float>* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::LDLT");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(csytrf)( &uplo, &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LDL^T, csytrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void LDLT
( char uplo, int n, std::complex<double>* A, int lda, int* ipiv,
  std::complex<double>* work, int lwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::LDLT");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(zsytrf)( &uplo, &n, A, &lda, ipiv, work, &lwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "LDL^T, zsytrf, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// Invert an LDL^T factorization                                              //
//----------------------------------------------------------------------------//

inline int InvertLDLTWorkSize( int n )
{
    return 2*n;
}

inline void InvertLDLT
( char uplo, int n, float* A, int lda, int* ipiv, float* work )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::InvertLDLT");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(ssytri)( &uplo, &n, A, &lda, ipiv, work, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLDL^T, ssytri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void InvertLDLT
( char uplo, int n, double* A, int lda, int* ipiv, double* work )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::InvertLDLT");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(dsytri)( &uplo, &n, A, &lda, ipiv, work, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLDL^T, dsytri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void InvertLDLT
( char uplo, int n, std::complex<float>* A, int lda, int* ipiv,
  std::complex<float>* work )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::InvertLDLT");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(csytri)( &uplo, &n, A, &lda, ipiv, work, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLDL^T, csytri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void InvertLDLT
( char uplo, int n, std::complex<double>* A, int lda, int* ipiv,
  std::complex<double>* work )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::InvertLDLT");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(zsytri)( &uplo, &n, A, &lda, ipiv, work, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "InvertLDL^T, zsytri, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

//----------------------------------------------------------------------------//
// EVD                                                                        //
//----------------------------------------------------------------------------//

inline int EVDWorkSize( int n )
{
    // Return 5 times the minimum recommendation
    return 5*std::max( 1, 1+6*n+2*n*n );
}

inline int EVDRealWorkSize( int n )
{
    return 5*std::max( 1, 1+5*n+2*n*n );
}

inline int EVDIntWorkSize( int n )
{
    return 5*std::max( 1, 3+5*n );
}

inline void EVD
( char jobz, char uplo, int n,
  float* A, int lda,
  float* w,
  float* work, int lwork,
  int* iwork, int liwork,
  __attribute__((unused)) float* rwork=0, __attribute__((unused)) int lrwork=0 )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::EVD");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(ssyevd)
    ( &jobz, &uplo, &n, A, &lda, w, work, &lwork, iwork, &liwork,
      &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "EVD, ssyevd, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void EVD
( char jobz, char uplo, int n,
  double* A, int lda,
  double* w,
  double* work, int lwork,
  int* iwork, int liwork,
  __attribute__((unused)) double* rwork=0, __attribute__((unused)) int lrwork=0 )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::EVD");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(dsyevd)
    ( &jobz, &uplo, &n, A, &lda, w, work, &lwork, iwork, &liwork,
      &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "EVD, dsyevd, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void EVD
( char jobz, char uplo, int n,
  std::complex<float>* A, int lda,
  float* w,
  std::complex<float>* work, int lwork,
  int* iwork, int liwork,
  float* rwork, int lrwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::EVD");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(cheevd)
    ( &jobz, &uplo, &n, A, &lda, w, work, &lwork, rwork, &lrwork,
      iwork, &liwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "EVD, cheevd, failed with info=" << info;
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void EVD
( char jobz, char uplo, int n,
  std::complex<double>* A, int lda,
  double* w,
  std::complex<double>* work, int lwork,
  int* iwork, int liwork,
  double* rwork, int lrwork )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::EVD");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
#endif
    int info;
    LAPACK(zheevd)
    ( &jobz, &uplo, &n, A, &lda, w, work, &lwork, rwork, &lrwork,
      iwork, &liwork, &info );
#ifndef RELEASE
    if( info != 0 )
    {
        std::ostringstream s;
        s << "EVD, zheevd, failed with info=" << info;
        std::cout << n << std::endl;
        for(int i=0; i<n; i++)
        {
            for(int j=0; j<n; j++)
                std::cout << A[i+j*n] << " ";
            std::cout << std::endl;
        }
        throw std::runtime_error( s.str().c_str() );
    }
#endif
}

inline void EVD
( char jobz, char uplo, int n, float* A, int lda, float* w )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::EVD");
#endif
    const int lwork = EVDWorkSize( n );
    const int liwork = EVDIntWorkSize( n );
    std::vector<float> work( lwork );
    std::vector<int> iwork( liwork );
    EVD( jobz, uplo, n, A, lda, w, &work[0], lwork, &iwork[0], liwork, 0, 0 );
}

inline void EVD
( char jobz, char uplo, int n, double* A, int lda, double* w )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::EVD");
#endif
    const int lwork = EVDWorkSize( n );
    const int liwork = EVDIntWorkSize( n );
    std::vector<double> work( lwork );
    std::vector<int> iwork( liwork );
    EVD( jobz, uplo, n, A, lda, w, &work[0], lwork, &iwork[0], liwork, 0, 0 );
}

inline void EVD
( char jobz, char uplo, int n, std::complex<float>* A, int lda, float* w )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::EVD");
#endif
    const int lwork = EVDWorkSize( n );
    const int liwork = EVDIntWorkSize( n );
    const int lrwork = EVDRealWorkSize( n );
    std::vector<std::complex<float> > work( lwork );
    std::vector<int> iwork( liwork );
    std::vector<float> rwork( lrwork );
    EVD
    ( jobz, uplo, n, A, lda, w,
      &work[0], lwork, &iwork[0], liwork, &rwork[0], lrwork );
}

inline void EVD
( char jobz, char uplo, int n, std::complex<double>* A, int lda, double* w )
{
#ifndef RELEASE
    CallStackEntry entry("lapack::EVD");
#endif
    const int lwork = EVDWorkSize( n );
    const int liwork = EVDIntWorkSize( n );
    const int lrwork = EVDRealWorkSize( n );
    std::vector<std::complex<double> > work( lwork );
    std::vector<int> iwork( liwork );
    std::vector<double> rwork( lrwork );
    EVD
    ( jobz, uplo, n, A, lda, w,
      &work[0], lwork, &iwork[0], liwork, &rwork[0], lrwork );
}

} // namespace lapack
} // namespace hifde3d

#endif // ifndef HIFDE3D_LAPACK_HPP

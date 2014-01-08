/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (HIFDE3D) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef HIFDE3D_BLAS_HPP
#define HIFDE3D_BLAS_HPP 1

#include "hifde3d/config.h"
#include <complex>
#include <sstream>
#include <stdexcept>

#if defined(BLAS_POST)
#define BLAS(name) name ## _
#else
#define BLAS(name) name
#endif

// TODO: Separate implementation from interface.

// Declarations for an external BLAS library
extern "C" {

typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;

float BLAS(snrm2)
( const int* n, const float* x, const int* incx );

double BLAS(dnrm2)
( const int* n, const double* x, const int* incx );

float BLAS(scnrm2)
( const int* n, const scomplex* x, const int* incx );

double BLAS(dznrm2)
( const int* n, const dcomplex* x, const int* incx );

void BLAS(sscal)
( const int* n, const float* alpha, float* x, const int* incx );

void BLAS(dscal)
( const int* n, const double* alpha, double* x, const int* incx );

void BLAS(cscal)
( const int* n, const scomplex* alpha, scomplex* x, const int* incx );

void BLAS(zscal)
( const int* n, const dcomplex* alpha, dcomplex* x, const int* incx );

void BLAS(sger)
( const int* m, const int* n,
  const float* alpha, const float* x, const int* incx,
                      const float* y, const int* incy,
                            float* A, const int* lda );

void BLAS(dger)
( const int* m, const int* n,
  const double* alpha, const double* x, const int* incx,
                       const double* y, const int* incy,
                             double* A, const int* lda );

void BLAS(cgerc)
( const int* m, const int* n,
  const scomplex* alpha, const scomplex* x, const int* incx,
                         const scomplex* y, const int* incy,
                               scomplex* A, const int* lda );

void BLAS(zgerc)
( const int* m, const int* n,
  const dcomplex* alpha, const dcomplex* x, const int* incx,
                         const dcomplex* y, const int* incy,
                               dcomplex* A, const int* lda );

void BLAS(sgemv)
( const char* trans, const int* m, const int* n,
  const float* alpha, const float* A, const int* lda,
                      const float* x, const int* incx,
  const float* beta,        float* y, const int* incy );

void BLAS(dgemv)
( const char* trans, const int* m, const int* n,
  const double* alpha, const double* A, const int* lda,
                       const double* x, const int* incx,
  const double* beta,        double* y, const int* incy );

void BLAS(cgemv)
( const char* trans, const int* m, const int* n,
  const scomplex* alpha, const scomplex* A, const int* lda,
                         const scomplex* x, const int* incx,
  const scomplex* beta,        scomplex* y, const int* incy );

void BLAS(zgemv)
( const char* trans, const int* m, const int* n,
  const dcomplex* alpha, const dcomplex* A, const int* lda,
                         const dcomplex* x, const int* incx,
  const dcomplex* beta,        dcomplex* y, const int* incy );

void BLAS(ssymv)
( const char* uplo, const int* n,
  const float* alpha, const float* A, const int* lda,
                      const float* x, const int* incx,
  const float* beta,        float* y, const int* incy );

void BLAS(dsymv)
( const char* uplo, const int* n,
  const double* alpha, const double* A, const int* lda,
                       const double* x, const int* incx,
  const double* beta,        double* y, const int* incy );

void BLAS(csymv)
( const char* uplo, const int* n,
  const scomplex* alpha, const scomplex* A, const int* lda,
                         const scomplex* x, const int* incx,
  const scomplex* beta,        scomplex* y, const int* incy );

void BLAS(zsymv)
( const char* uplo, const int* n,
  const dcomplex* alpha, const dcomplex* A, const int* lda,
                         const dcomplex* x, const int* incx,
  const dcomplex* beta,        dcomplex* y, const int* incy );

void BLAS(sgemm)
( const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const float* alpha, const float* A, const int* lda,
                      const float* B, const int* ldb,
  const float* beta,        float* C, const int* ldc );

void BLAS(dgemm)
( const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const double* alpha, const double* A, const int* lda,
                       const double* B, const int* ldb,
  const double* beta,        double* C, const int* ldc );

void BLAS(cgemm)
( const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const scomplex* alpha, const scomplex* A, const int* lda,
                         const scomplex* B, const int* ldb,
  const scomplex* beta,        scomplex* C, const int* ldc );

void BLAS(zgemm)
( const char* transa, const char* transb,
  const int* m, const int* n, const int* k,
  const dcomplex* alpha, const dcomplex* A, const int* lda,
                         const dcomplex* B, const int* ldb,
  const dcomplex* beta,        dcomplex* C, const int* ldc );

void BLAS(ssymm)
( const char* side, const char* uplo, const int* m, const int* n,
  const float* alpha, const float* A, const int* lda,
                      const float* B, const int* ldb,
  const float* beta,        float* C, const int* ldc );

void BLAS(dsymm)
( const char* side, const char* uplo, const int* m, const int* n,
  const double* alpha, const double* A, const int* lda,
                       const double* B, const int* ldb,
  const double* beta,        double* C, const int* ldc );

void BLAS(csymm)
( const char* side, const char* uplo, const int* m, const int* n,
  const scomplex* alpha, const scomplex* A, const int* lda,
                         const scomplex* B, const int* ldb,
  const scomplex* beta,        scomplex* C, const int* ldc );

void BLAS(zsymm)
( const char* side, const char* uplo, const int* m, const int* n,
  const dcomplex* alpha, const dcomplex* A, const int* lda,
                         const dcomplex* B, const int* ldb,
  const dcomplex* beta,        dcomplex* C, const int* ldc );

void BLAS(strmm)
( const char* side, const char* uplo, const char* trans, const char* diag,
  const int* m, const int* n,
  const float* alpha, const float* A, const int* lda,
                            float* B, const int* ldb );

void BLAS(dtrmm)
( const char* side, const char* uplo, const char* trans, const char* diag,
  const int* m, const int* n,
  const double* alpha, const double* A, const int* lda,
                             double* B, const int* ldb );

void BLAS(ctrmm)
( const char* side, const char* uplo, const char* trans, const char* diag,
  const int* m, const int* n,
  const scomplex* alpha, const scomplex* A, const int* lda,
                               scomplex* B, const int* ldb );

void BLAS(ztrmm)
( const char* side, const char* uplo, const char* trans, const char* diag,
  const int* m, const int* n,
  const dcomplex* alpha, const dcomplex* A, const int* lda,
                               dcomplex* B, const int* ldb );

} // extern "C"

namespace hifde3d {
namespace blas {

//----------------------------------------------------------------------------//
// Compute the two-norm of a vector                                           //
//----------------------------------------------------------------------------//

inline float Nrm2
( int n, const float* x, int incx )
{ return BLAS(snrm2)( &n, x, &incx ); }

inline double Nrm2
( int n, const double* x, int incx )
{ return BLAS(dnrm2)( &n, x, &incx ); }

inline float Nrm2
( int n, const scomplex* x, int incx )
{ return BLAS(scnrm2)( &n, x, &incx ); }

inline double Nrm2
( int n, const dcomplex* x, int incx )
{ return BLAS(dznrm2)( &n, x, &incx ); }

//----------------------------------------------------------------------------//
// Scale a vector                                                             //
//----------------------------------------------------------------------------//

inline void Scal
( int n, float alpha, float* x, int incx )
{ BLAS(sscal)( &n, &alpha, x, &incx ); }

inline void Scal
( int n, double alpha, double* x, int incx )
{ BLAS(dscal)( &n, &alpha, x, &incx ); }

inline void Scal
( int n, std::complex<float> alpha, std::complex<float>* x, int incx )
{ BLAS(cscal)( &n, &alpha, x, &incx ); }

inline void Scal
( int n, std::complex<double> alpha, std::complex<double>* x, int incx )
{ BLAS(zscal)( &n, &alpha, x, &incx ); }

//----------------------------------------------------------------------------//
// General rank-one update                                                    //
//----------------------------------------------------------------------------//

inline void Ger
( int m, int n,
  float alpha, const float* x, int incx,
               const float* y, int incy,
                     float* A, int lda )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Ger");
    if( lda == 0 )
        throw std::logic_error("lda = 0");
#endif
    BLAS(sger)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda );
}

inline void Ger
( int m, int n,
  double alpha, const double* x, int incx,
                const double* y, int incy,
                      double* A, int lda )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Ger");
    if( lda == 0 )
        throw std::logic_error("lda = 0");
#endif
    BLAS(dger)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda );
}

inline void Ger
( int m, int n,
  std::complex<float> alpha, const std::complex<float>* x, int incx,
                             const std::complex<float>* y, int incy,
                                   std::complex<float>* A, int lda )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Ger");
    if( lda == 0 )
        throw std::logic_error("lda = 0");
#endif
    BLAS(cgerc)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda );
}

inline void Ger
( int m, int n,
  std::complex<double> alpha, const std::complex<double>* x, int incx,
                              const std::complex<double>* y, int incy,
                                    std::complex<double>* A, int lda )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Ger");
    if( lda == 0 )
        throw std::logic_error("lda = 0");
#endif
    BLAS(zgerc)( &m, &n, &alpha, x, &incx, y, &incy, A, &lda );
}

//----------------------------------------------------------------------------//
// General matrix-vector multiplication                                       //
//----------------------------------------------------------------------------//

inline void Gemv
( char trans, int m, int n,
  float alpha, const float* A, int lda,
               const float* x, int incx,
  float beta,        float* y, int incy )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Gemv");
    if( lda == 0 )
        throw std::logic_error("lda = 0");
#endif
    if( trans == 'N' )
    {
        if( n != 0 )
            BLAS(sgemv)
            ( &trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
        else
        {
            if( beta == 0 )
            {
                if( incy == 1 )
                    std::memset( y, 0, m*sizeof(float) );
                else
                    for( int i=0; i<m; ++i )
                        y[i*incy] = 0;
            }
            else
                for( int i=0; i<m; ++i )
                    y[i*incy] *= beta;
        }
    }
    else
    {
        if( m != 0 )
            BLAS(sgemv)
            ( &trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
        else
        {
            if( beta == 0 )
            {
                if( incy == 1 )
                    std::memset( y, 0, n*sizeof(float) );
                else
                    for( int i=0; i<n; ++i )
                        y[i*incy] = 0;
            }
            else
                for( int i=0; i<n; ++i )
                    y[i*incy] *= beta;
        }
    }
}

inline void Gemv
( char trans, int m, int n,
  double alpha, const double* A, int lda,
                const double* x, int incx,
  double beta,        double* y, int incy )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Gemv");
    if( lda == 0 )
        throw std::logic_error("lda = 0");
#endif
    if( trans == 'N' )
    {
        if( n != 0 )
            BLAS(dgemv)
            ( &trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
        else
        {
            if( beta == 0 )
            {
                if( incy == 1 )
                    std::memset( y, 0, m*sizeof(double) );
                else
                    for( int i=0; i<m; ++i )
                        y[i*incy] = 0;
            }
            else
                for( int i=0; i<m; ++i )
                    y[i*incy] *= beta;
        }
    }
    else
    {
        if( m != 0 )
            BLAS(dgemv)
            ( &trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
        else
        {
            if( beta == 0 )
            {
                if( incy == 1 )
                    std::memset( y, 0, n*sizeof(double) );
                else
                    for( int i=0; i<n; ++i )
                        y[i*incy] = 0;
            }
            else
                for( int i=0; i<n; ++i )
                    y[i*incy] *= beta;
        }
    }
}

inline void Gemv
( char trans, int m, int n,
  std::complex<float> alpha, const std::complex<float>* A, int lda,
                             const std::complex<float>* x, int incx,
  std::complex<float> beta,        std::complex<float>* y, int incy )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Gemv");
    if( lda == 0 )
        throw std::logic_error("lda = 0");
#endif
    if( trans == 'N' )
    {
        if( n != 0 )
            BLAS(cgemv)
            ( &trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
        else
        {
            if( beta == std::complex<float>(0) )
            {
                if( incy == 1 )
                    std::memset( y, 0, m*sizeof(std::complex<float>) );
                else
                    for( int i=0; i<m; ++i )
                        y[i*incy] = 0;
            }
            else
                for( int i=0; i<m; ++i )
                    y[i*incy] *= beta;
        }
    }
    else
    {
        if( m != 0 )
            BLAS(cgemv)
            ( &trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
        else
        {
            if( beta == std::complex<float>(0) )
            {
                if( incy == 1 )
                    std::memset( y, 0, n*sizeof(std::complex<float>) );
                else
                    for( int i=0; i<n; ++i )
                        y[i*incy] = 0;
            }
            else
                for( int i=0; i<n; ++i )
                    y[i*incy] *= beta;
        }
    }
}

inline void Gemv
( char trans, int m, int n,
  std::complex<double> alpha, const std::complex<double>* A, int lda,
                              const std::complex<double>* x, int incx,
  std::complex<double> beta,        std::complex<double>* y, int incy )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Gemv");
    if( lda == 0 )
        throw std::logic_error("lda = 0");
#endif
    if( trans == 'N' )
    {
        if( n != 0 )
            BLAS(zgemv)
            ( &trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
        else
        {
            if( beta == std::complex<double>(0) )
            {
                if( incy == 1 )
                    std::memset( y, 0, m*sizeof(std::complex<double>) );
                else
                    for( int i=0; i<m; ++i )
                        y[i*incy] = 0;
            }
            else
                for( int i=0; i<m; ++i )
                    y[i*incy] *= beta;
        }
    }
    else
    {
        if( m != 0 )
            BLAS(zgemv)
            ( &trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
        else
        {
            if( beta == std::complex<double>(0) )
            {
                if( incy == 1 )
                    std::memset( y, 0, n*sizeof(std::complex<double>) );
                else
                    for( int i=0; i<n; ++i )
                        y[i*incy] = 0;
            }
            else
                for( int i=0; i<n; ++i )
                    y[i*incy] *= beta;
        }
    }
}

//----------------------------------------------------------------------------//
// Symmetric matrix-vector multiplication                                     //
//----------------------------------------------------------------------------//

inline void Symv
( char uplo, int n,
  float alpha, const float* A, int lda,
               const float* x, int incx,
  float beta,        float* y, int incy )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Symv");
    if( lda == 0 )
        throw std::logic_error("lda = 0");
#endif
    BLAS(ssymv)( &uplo, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

inline void Symv
( char uplo, int n,
  double alpha, const double* A, int lda,
                const double* x, int incx,
  double beta,        double* y, int incy )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Symv");
    if( lda == 0 )
        throw std::logic_error("lda = 0");
#endif
    BLAS(dsymv)( &uplo, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

inline void Symv
( char uplo, int n,
  std::complex<float> alpha, const std::complex<float>* A, int lda,
                             const std::complex<float>* x, int incx,
  std::complex<float> beta,        std::complex<float>* y, int incy )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Symv");
    if( lda == 0 )
        throw std::logic_error("lda = 0");
#endif
    BLAS(csymv)( &uplo, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

inline void Symv
( char uplo, int n,
  std::complex<double> alpha, const std::complex<double>* A, int lda,
                              const std::complex<double>* x, int incx,
  std::complex<double> beta,        std::complex<double>* y, int incy )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Symv");
    if( lda == 0 )
        throw std::logic_error("lda = 0");
#endif
    BLAS(zsymv)( &uplo, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

//----------------------------------------------------------------------------//
// General matrix-matrix multiplication                                       //
//----------------------------------------------------------------------------//

inline void PrintGemmDimensions
( int m, int n, int k, int lda, int ldb, int ldc, std::ostream& os=std::cout )
{
    os << "(m,n,k)=" << m << "," << n << "," << k << ", (lda,ldb,ldc)="
       << lda << "," << ldb << "," << ldc << std::endl;
}

inline void Gemm
( char transa, char transb, int m, int n, int k,
  float alpha, const float* A, int lda,
               const float* B, int ldb,
  float beta,        float* C, int ldc )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Gemm");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldb == 0 )
        throw std::logic_error("ldb was 0");
    if( ldc == 0 )
        throw std::logic_error("ldc was 0");
    if( transa == 'N' && lda < m )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid lda for non-transposed A");
    }
    else if( (transa =='C' || transa=='T') && lda < k )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid lda for transposed A");
    }
    if( transb == 'N' && ldb < k )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid ldb for non-transposed B");
    }
    else if( (transb == 'C' || transb == 'T') && ldb < n )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid ldb for transposed B");
    }
    if( ldc < m )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid ldc for C");
    }
#endif
    if( k != 0 )
        BLAS(sgemm)
        ( &transa, &transb, &m, &n, &k,
          &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
    else
    {
        if( beta == 0 )
            for( int j=0; j<n; ++j )
                std::memset( &C[j*ldc], 0, m*sizeof(float) );
        else
            for( int j=0; j<n; ++j )
                for( int i=0; i<m; ++i )
                    C[j*ldc+i] *= beta;
    }
}

inline void Gemm
( char transa, char transb, int m, int n, int k,
  double alpha, const double* A, int lda,
                const double* B, int ldb,
  double beta,        double* C, int ldc )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Gemm");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldb == 0 )
        throw std::logic_error("ldb was 0");
    if( ldc == 0 )
        throw std::logic_error("ldc was 0");
    if( transa == 'N' && lda < m )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid lda for non-transposed A");
    }
    else if( (transa =='C' || transa=='T') && lda < k )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid lda for transposed A");
    }
    if( transb == 'N' && ldb < k )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid ldb for non-transposed B");
    }
    else if( (transb == 'C' || transb == 'T') && ldb < n )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid ldb for transposed B");
    }
    if( ldc < m )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid ldc for C");
    }
#endif
    if( k != 0 )
        BLAS(dgemm)
        ( &transa, &transb, &m, &n, &k,
          &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
    else
    {
        if( beta == 0 )
            for( int j=0; j<n; ++j )
                std::memset( &C[j*ldc], 0, m*sizeof(double) );
        else
            for( int j=0; j<n; ++j )
                for( int i=0; i<m; ++i )
                    C[j*ldc+i] *= beta;
    }
}

inline void Gemm
( char transa, char transb, int m, int n, int k,
  std::complex<float> alpha, const std::complex<float>* A, int lda,
                             const std::complex<float>* B, int ldb,
  std::complex<float> beta,        std::complex<float>* C, int ldc )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Gemm");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldb == 0 )
        throw std::logic_error("ldb was 0");
    if( ldc == 0 )
        throw std::logic_error("ldc was 0");
    if( transa == 'N' && lda < m )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid lda for non-transposed A");
    }
    else if( (transa =='C' || transa=='T') && lda < k )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid lda for transposed A");
    }
    if( transb == 'N' && ldb < k )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid ldb for non-transposed B");
    }
    else if( (transb == 'C' || transb == 'T') && ldb < n )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid ldb for transposed B");
    }
    if( ldc < m )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid ldc for C");
    }
#endif
    if( k != 0 )
        BLAS(cgemm)
        ( &transa, &transb, &m, &n, &k,
          &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
    else
    {
        if( beta == std::complex<float>(0) )
            for( int j=0; j<n; ++j )
                std::memset( &C[j*ldc], 0, m*sizeof(std::complex<float>) );
        else
            for( int j=0; j<n; ++j )
                for( int i=0; i<m; ++i )
                    C[j*ldc+i] *= beta;
    }
}

inline void Gemm
( char transa, char transb, int m, int n, int k,
  std::complex<double> alpha, const std::complex<double>* A, int lda,
                              const std::complex<double>* B, int ldb,
  std::complex<double> beta,        std::complex<double>* C, int ldc )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Gemm");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldb == 0 )
        throw std::logic_error("ldb was 0");
    if( ldc == 0 )
        throw std::logic_error("ldc was 0");
    if( transa == 'N' && lda < m )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid lda for non-transposed A");
    }
    else if( (transa =='C' || transa=='T') && lda < k )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid lda for transposed A");
    }
    if( transb == 'N' && ldb < k )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid ldb for non-transposed B");
    }
    else if( (transb == 'C' || transb == 'T') && ldb < n )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid ldb for transposed B");
    }
    if( ldc < m )
    {
        PrintGemmDimensions( m, n, k, lda, ldb, ldc );
        throw std::logic_error("Invalid ldc for C");
    }
#endif
    if( k != 0 )
        BLAS(zgemm)
        ( &transa, &transb, &m, &n, &k,
          &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
    else
    {
        if( beta == std::complex<double>(0) )
            for( int j=0; j<n; ++j )
                std::memset( &C[j*ldc], 0, m*sizeof(std::complex<double>) );
        else
            for( int j=0; j<n; ++j )
                for( int i=0; i<m; ++i )
                    C[j*ldc+i] *= beta;
    }
}

//----------------------------------------------------------------------------//
// Symmetric matrix-matrix multiplication                                     //
//----------------------------------------------------------------------------//

inline void Symm
( char side, char uplo, int m, int n,
  float alpha, const float* A, int lda,
               const float* B, int ldb,
  float beta,        float* C, int ldc )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Symm");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldb == 0 )
        throw std::logic_error("ldb was 0");
    if( ldc == 0 )
        throw std::logic_error("ldc was 0");
#endif
    BLAS(ssymm)
    ( &side, &uplo, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

inline void Symm
( char side, char uplo, int m, int n,
  double alpha, const double* A, int lda,
                const double* B, int ldb,
  double beta,        double* C, int ldc )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Symm");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldb == 0 )
        throw std::logic_error("ldb was 0");
    if( ldc == 0 )
        throw std::logic_error("ldc was 0");
#endif
    BLAS(dsymm)
    ( &side, &uplo, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

inline void Symm
( char side, char uplo, int m, int n,
  std::complex<float> alpha, const std::complex<float>* A, int lda,
                             const std::complex<float>* B, int ldb,
  std::complex<float> beta,        std::complex<float>* C, int ldc )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Symm");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldb == 0 )
        throw std::logic_error("ldb was 0");
    if( ldc == 0 )
        throw std::logic_error("ldc was 0");
#endif
    BLAS(csymm)
    ( &side, &uplo, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

inline void Symm
( char side, char uplo, int m, int n,
  std::complex<double> alpha, const std::complex<double>* A, int lda,
                              const std::complex<double>* B, int ldb,
  std::complex<double> beta,        std::complex<double>* C, int ldc )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Symm");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldb == 0 )
        throw std::logic_error("ldb was 0");
    if( ldc == 0 )
        throw std::logic_error("ldc was 0");
#endif
    BLAS(zsymm)
    ( &side, &uplo, &m, &n, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

//----------------------------------------------------------------------------//
// Triangular matrix-matrix multiplication                                    //
//----------------------------------------------------------------------------//

inline void Trmm
( char side, char uplo, char transa, char diag, int m, int n,
  float alpha, const float* A, int lda,
                     float* B, int ldb )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Trmm");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldb == 0 )
        throw std::logic_error("ldb was 0");
#endif
    BLAS(strmm)
    ( &side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B, &ldb );
}

inline void Trmm
( char side, char uplo, char transa, char diag, int m, int n,
  double alpha, const double* A, int lda,
                      double* B, int ldb )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Trmm");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldb == 0 )
        throw std::logic_error("ldb was 0");
#endif
    BLAS(dtrmm)
    ( &side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B, &ldb );
}

inline void Trmm
( char side, char uplo, char transa, char diag, int m, int n,
  std::complex<float> alpha, const std::complex<float>* A, int lda,
                                   std::complex<float>* B, int ldb )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Trmm");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldb == 0 )
        throw std::logic_error("ldb was 0");
#endif
    BLAS(ctrmm)
    ( &side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B, &ldb );
}

inline void Trmm
( char side, char uplo, char transa, char diag, int m, int n,
  std::complex<double> alpha, const std::complex<double>* A, int lda,
                                    std::complex<double>* B, int ldb )
{
#ifndef RELEASE
    CallStackEntry entry("blas::Trmm");
    if( lda == 0 )
        throw std::logic_error("lda was 0");
    if( ldb == 0 )
        throw std::logic_error("ldb was 0");
#endif
    BLAS(ztrmm)
    ( &side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B, &ldb );
}

} // namespace blas
} // namespace hifde3d

#endif // ifndef HIFDE3D_BLAS_HPP

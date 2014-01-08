/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (HIFDE3D) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "hifde3d.hpp"

namespace hifde3d {
namespace hmat_tools {

// Dense y := alpha A^H x + beta y
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::AdjointMultiply (y := D^H x + y)");
#endif
    if( A.Symmetric() )
    {
        Vector<Scalar> xConj;
        Conjugate( x, xConj );
        Conjugate( y );
        blas::Symv
        ( 'L', A.Height(),
          Conj(alpha), A.LockedBuffer(), A.LDim(),
                       xConj.Buffer(),   1,
          Conj(beta),  y.Buffer(),       1 );
        Conjugate( y );
    }
    else
    {
        blas::Gemv
        ( 'C', A.Height(), A.Width(),
          alpha, A.LockedBuffer(), A.LDim(),
                 x.LockedBuffer(), 1,
          beta,  y.Buffer(),       1 );
    }
}

// Dense y := alpha A^H x
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Vector<Scalar>& x,
                      Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::AdjointMultiply (y := D^H x)");
#endif
    y.Resize( A.Width() );
    if( A.Symmetric() )
    {
        Vector<Scalar> xConj;
        Conjugate( x, xConj );
        blas::Symv
        ( 'L', A.Height(),
          Conj(alpha), A.LockedBuffer(), A.LDim(),
                       xConj.Buffer(),   1,
          0,           y.Buffer(),       1 );
        Conjugate( y );
    }
    else
    {
        blas::Gemv
        ( 'C', A.Height(), A.Width(),
          alpha, A.LockedBuffer(), A.LDim(),
                 x.LockedBuffer(), 1,
          0,     y.Buffer(),       1 );
    }
}

// Low-rank y := alpha A^H x + beta y
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const LowRank<Scalar>& A,
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::AdjointMultiply (y := F x + y)");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int r = A.Rank();

    // Form t := alpha (A.U)^H x
    Vector<Scalar> t( r );
    blas::Gemv
    ( 'C', m, r,
      alpha, A.U.LockedBuffer(), A.U.LDim(),
             x.LockedBuffer(),   1,
      0,     t.Buffer(),         1 );

    Conjugate( t );
    Conjugate( y );
    blas::Gemv
    ( 'N', n, r,
      1,          A.V.LockedBuffer(), A.V.LDim(),
                  t.LockedBuffer(),   1,
      Conj(beta), y.Buffer(),         1 );
    Conjugate( y );
}

// Low-rank y := alpha A^H x
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const LowRank<Scalar>& A,
                const Vector<Scalar>& x,
                      Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::AdjointMultiply (y := F x)");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int r = A.Rank();

    // Form t := alpha (A.U)^H x
    Vector<Scalar> t( r );
    blas::Gemv
    ( 'C', m, r,
      alpha, A.U.LockedBuffer(), A.U.LDim(),
             x.LockedBuffer(),   1,
      0,     t.Buffer(),         1 );

    y.Resize( n );
    Conjugate( t );
    blas::Gemv
    ( 'N', n, r,
      1, A.V.LockedBuffer(), A.V.LDim(),
         t.LockedBuffer(),   1,
      0, y.Buffer(),         1 );
    Conjugate( y );
}

template void AdjointMultiply
( float alpha, const Dense<float>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void AdjointMultiply
( double alpha, const Dense<double>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void AdjointMultiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Vector<std::complex<float> >& x,
  std::complex<float> beta,        Vector<std::complex<float> >& y );
template void AdjointMultiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Vector<std::complex<double> >& x,
  std::complex<double> beta,        Vector<std::complex<double> >& y );

template void AdjointMultiply
( float alpha, const Dense<float>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void AdjointMultiply
( double alpha, const Dense<double>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void AdjointMultiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Vector<std::complex<float> >& x,
                                   Vector<std::complex<float> >& y );
template void AdjointMultiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Vector<std::complex<double> >& x,
                                    Vector<std::complex<double> >& y );

template void AdjointMultiply
( float alpha, const LowRank<float>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void AdjointMultiply
( double alpha, const LowRank<double>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void AdjointMultiply
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const Vector<std::complex<float> >& x,
  std::complex<float> beta,
        Vector<std::complex<float> >& y );
template void AdjointMultiply
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const Vector<std::complex<double> >& x,
  std::complex<double> beta,
        Vector<std::complex<double> >& y );

template void AdjointMultiply
( float alpha, const LowRank<float>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void AdjointMultiply
( double alpha, const LowRank<double>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void AdjointMultiply
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const Vector<std::complex<float> >& x,
        Vector<std::complex<float> >& y );
template void AdjointMultiply
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const Vector<std::complex<double> >& x,
        Vector<std::complex<double> >& y );

} // namespace hmat_tools
} // namespace hifde3d

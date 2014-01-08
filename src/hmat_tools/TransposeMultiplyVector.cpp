/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (HIFDE3D) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "hifde3d/core/environment.hpp"
#include "hifde3d/core/hmat_tools.hpp"

namespace hifde3d {
namespace hmat_tools {

// Dense y := alpha A^T x + beta y
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::TransposeMultiply (y := D^T x + y)");
#endif
    if( A.Symmetric() )
    {
        blas::Symv
        ( 'L', A.Height(),
          alpha, A.LockedBuffer(), A.LDim(),
                 x.LockedBuffer(), 1,
          beta,  y.Buffer(),       1 );
    }
    else
    {
        blas::Gemv
        ( 'T', A.Height(), A.Width(),
          alpha, A.LockedBuffer(), A.LDim(),
                 x.LockedBuffer(), 1,
          beta,  y.Buffer(),       1 );
    }
}

// Dense y := alpha A^T x
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Vector<Scalar>& x,
                      Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::TransposeMultiply (y := D^T x)");
#endif
    y.Resize( A.Width() );
    if( A.Symmetric() )
    {
        blas::Symv
        ( 'L', A.Height(),
          alpha, A.LockedBuffer(), A.LDim(),
                 x.LockedBuffer(), 1,
          0,     y.Buffer(),       1 );
    }
    else
    {
        blas::Gemv
        ( 'T', A.Height(), A.Width(),
          alpha, A.LockedBuffer(), A.LDim(),
                 x.LockedBuffer(), 1,
          0,     y.Buffer(),       1 );
    }
}

// Low-rank y := alpha A^T x + beta y
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const LowRank<Scalar>& A,
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::TransposeMultiply (y := F^T x + y)");
#endif
    const int r = A.Rank();

    // Form t := alpha (A.U)^T x
    Vector<Scalar> t( r );
    blas::Gemv
    ( 'T', A.Height(), r,
      alpha, A.U.LockedBuffer(), A.U.LDim(),
             x.LockedBuffer(),   1,
      0,     t.Buffer(),         1 );

    // Form y := (A.V) t + beta y
    blas::Gemv
    ( 'N', A.Width(), r,
      1,    A.V.LockedBuffer(), A.V.LDim(),
            t.LockedBuffer(),   1,
      beta, y.Buffer(),         1 );
}

// Low-rank y := alpha A^T x
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const LowRank<Scalar>& A,
                const Vector<Scalar>& x,
                      Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::TransposeMultiply (y := F^T x)");
#endif
    const int r = A.Rank();

    // Form t := alpha (A.U)^T x
    Vector<Scalar> t( r );
    blas::Gemv
    ( 'T', A.Height(), r,
      alpha, A.U.LockedBuffer(), A.U.LDim(),
             x.LockedBuffer(),   1,
      0,     t.Buffer(),         1 );

    y.Resize( A.Width() );
    // Form y := (A.V) t
    blas::Gemv
    ( 'N', A.Width(), r,
      1, A.V.LockedBuffer(), A.V.LDim(),
         t.LockedBuffer(),   1,
      0, y.Buffer(),         1 );
}

template void TransposeMultiply
( float alpha, const Dense<float>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void TransposeMultiply
( double alpha, const Dense<double>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void TransposeMultiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Vector<std::complex<float> >& x,
  std::complex<float> beta,        Vector<std::complex<float> >& y );
template void TransposeMultiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Vector<std::complex<double> >& x,
  std::complex<double> beta,        Vector<std::complex<double> >& y );

template void TransposeMultiply
( float alpha, const Dense<float>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void TransposeMultiply
( double alpha, const Dense<double>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void TransposeMultiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Vector<std::complex<float> >& x,
                                   Vector<std::complex<float> >& y );
template void TransposeMultiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Vector<std::complex<double> >& x,
                                    Vector<std::complex<double> >& y );

template void TransposeMultiply
( float alpha, const LowRank<float>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void TransposeMultiply
( double alpha, const LowRank<double>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void TransposeMultiply
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const Vector<std::complex<float> >& x,
  std::complex<float> beta,
        Vector<std::complex<float> >& y );
template void TransposeMultiply
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const Vector<std::complex<double> >& x,
  std::complex<double> beta,
        Vector<std::complex<double> >& y );

template void TransposeMultiply
( float alpha, const LowRank<float>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void TransposeMultiply
( double alpha, const LowRank<double>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void TransposeMultiply
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const Vector<std::complex<float> >& x,
        Vector<std::complex<float> >& y );
template void TransposeMultiply
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const Vector<std::complex<double> >& x,
        Vector<std::complex<double> >& y );

} // namespace hmat_tools
} // namespace hifde3d

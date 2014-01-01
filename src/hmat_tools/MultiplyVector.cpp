#include "HIFDE3D.hpp"

// Dense y := alpha A x + beta y
template<typename Scalar>
void Multiply
( Scalar alpha, const Dense<Scalar>& A,
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Multiply (y := D x + y)");
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
        ( 'N', A.Height(), A.Width(),
          alpha, A.LockedBuffer(), A.LDim(),
                 x.LockedBuffer(), 1,
          beta,  y.Buffer(),       1 );
    }
}

// Dense y := alpha A x
template<typename Scalar>
void Multiply
( Scalar alpha, const Dense<Scalar>& A,
                const Vector<Scalar>& x,
                      Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Multiply (y := D x)");
#endif
    y.Resize( A.Height() );
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
        ( 'N', A.Height(), A.Width(),
          alpha, A.LockedBuffer(), A.LDim(),
                 x.LockedBuffer(), 1,
          0,     y.Buffer(),       1 );
    }
}

// Low-rank y := alpha A x + beta y
template<typename Scalar>
void Multiply
( Scalar alpha, const LowRank<Scalar>& A,
                const Vector<Scalar>& x,
  Scalar beta,        Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Multiply (y := F x + y)");
#endif
    const int r = A.Rank();

    // Form t := alpha (A.V)^[T,H] x
    Vector<Scalar> t( r );
    const char option = 'T';
    blas::Gemv
    ( option, A.Width(), r,
      alpha, A.V.LockedBuffer(), A.V.LDim(),
             x.LockedBuffer(),   1,
      0,     t.Buffer(),         1 );

    // Form y := (A.U) t + beta y
    blas::Gemv
    ( 'N', A.Height(), r,
      1,    A.U.LockedBuffer(), A.U.LDim(),
            t.LockedBuffer(),   1,
      beta, y.Buffer(),         1 );
}

// Low-rank y := alpha A x
template<typename Scalar>
void Multiply
( Scalar alpha, const LowRank<Scalar>& A,
                const Vector<Scalar>& x,
                      Vector<Scalar>& y )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Multiply (y := F x)");
#endif
    const int r = A.Rank();

    // Form t := alpha (A.V)^[T,H] x
    Vector<Scalar> t( r );
    const char option = 'T';
    blas::Gemv
    ( option, A.Width(), r,
      alpha, A.V.LockedBuffer(), A.V.LDim(),
             x.LockedBuffer(),   1,
      0,     t.Buffer(),         1 );

    // Form y := (A.U) t
    y.Resize( A.Height() );
    blas::Gemv
    ( 'N', A.Height(), r,
      1, A.U.LockedBuffer(), A.U.LDim(),
         t.LockedBuffer(),   1,
      0, y.Buffer(),         1 );
}

template void Multiply
( float alpha, const Dense<float>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void Multiply
( double alpha, const Dense<double>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void Multiply
( std::complex<float> alpha, const Dense< std::complex<float> >& A,
                             const Vector<std::complex<float> >& x,
  std::complex<float> beta,        Vector<std::complex<float> >& y );
template void Multiply
( std::complex<double> alpha, const Dense< std::complex<double> >& A,
                              const Vector<std::complex<double> >& x,
  std::complex<double> beta,        Vector<std::complex<double> >& y );

template void Multiply
( float alpha, const Dense<float>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void Multiply
( double alpha, const Dense<double>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void Multiply
( std::complex<float> alpha, const Dense< std::complex<float> >& A,
                             const Vector<std::complex<float> >& x,
                                   Vector<std::complex<float> >& y );
template void Multiply
( std::complex<double> alpha, const Dense< std::complex<double> >& A,
                              const Vector<std::complex<double> >& x,
                                    Vector<std::complex<double> >& y );

template void Multiply
( float alpha, const LowRank<float>& A,
               const Vector<float>& x,
  float beta,        Vector<float>& y );
template void Multiply
( double alpha, const LowRank<double>& A,
                const Vector<double>& x,
  double beta,        Vector<double>& y );
template void Multiply
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const Vector<std::complex<float> >& x,
  std::complex<float> beta,
        Vector<std::complex<float> >& y );
template void Multiply
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const Vector<std::complex<double> >& x,
  std::complex<double> beta,
        Vector<std::complex<double> >& y );

template void Multiply
( float alpha, const LowRank<float>& A,
               const Vector<float>& x,
                     Vector<float>& y );
template void Multiply
( double alpha, const LowRank<double>& A,
                const Vector<double>& x,
                      Vector<double>& y );
template void Multiply
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const Vector<std::complex<float> >& x,
        Vector<std::complex<float> >& y );
template void Multiply
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const Vector<std::complex<double> >& x,
        Vector<std::complex<double> >& y );



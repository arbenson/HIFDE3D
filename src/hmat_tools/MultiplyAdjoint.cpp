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

// Dense C := alpha A B^H
template<typename Scalar>
void MultiplyAdjoint
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyAdjoint (D := D D^H)");
#endif
    C.SetType( GENERAL );
    C.Resize( A.Height(), B.Height() );
    MultiplyAdjoint( alpha, A, B, Scalar(0), C );
}

// Dense C := alpha A B^H + beta C
template<typename Scalar>
void MultiplyAdjoint
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry
    ("hmat_tools::MultiplyAdjoint (D := D D^H + D)");
    if( A.Width() != B.Width() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( A.Height() != C.Height() )
        throw std::logic_error("The height of A and C are nonconformal.");
    if( B.Height() != C.Width() )
        throw std::logic_error("The width of B and C are nonconformal.");
    if( B.Symmetric() || A.Symmetric() )
        throw std::logic_error("BLAS does not support symm times trans.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    blas::Gemm
    ( 'N', 'C', C.Height(), C.Width(), A.Width(),
      alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(),
      beta, C.Buffer(), C.LDim() );
}

// TODO: version of above routine that allows for temporary in-place conj of B

// Form a dense matrix from a dense matrix times a low-rank matrix
template<typename Scalar>
void MultiplyAdjoint
( Scalar alpha, const Dense<Scalar>& A,
                const LowRank<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyAdjoint (D := D F^H)");
#endif
    C.SetType( GENERAL );
    C.Resize( A.Height(), B.Height() );
    MultiplyAdjoint( alpha, A, B, Scalar(0), C );
}

// Form a dense matrix from a dense matrix times a low-rank matrix
template<typename Scalar>
void MultiplyAdjoint
( Scalar alpha, const Dense<Scalar>& A,
                const LowRank<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry
    ("hmat_tools::MultiplyAdjoint (D := D F^H + D)");
    if( A.Width() != B.Width() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( A.Height() != C.Height() )
        throw std::logic_error("The height of A and C are nonconformal.");
    if( B.Height() != C.Width() )
        throw std::logic_error("The width of B and C are nonconformal.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    const int m = C.Height();
    const int n = C.Width();
    const int r = B.Rank();

    // C := alpha (A conj(B.V)) B.U^H + beta C
    //
    // W := A conj(B.V)
    // C := alpha W B.U^H + beta C

    Dense<Scalar> W( m, r ), cBV;
    Conjugate( B.V, cBV );
    blas::Gemm
    ( 'N', 'N', m, r, B.Width(),
      1, A.LockedBuffer(),   A.LDim(),
         cBV.LockedBuffer(), cBV.LDim(),
      0, W.Buffer(),         W.LDim() );
    blas::Gemm
    ( 'N', 'C', m, n, r,
      alpha, W.LockedBuffer(),   W.LDim(),
             B.U.LockedBuffer(), B.U.LDim(),
      beta,  C.Buffer(),         C.LDim() );
}

// Form a dense matrix from a low-rank matrix times a dense matrix
template<typename Scalar>
void MultiplyAdjoint
( Scalar alpha, const LowRank<Scalar>& A,
                const Dense<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyAdjoint (D := F D^H)");
#endif
    C.SetType( GENERAL );
    C.Resize( A.Height(), B.Height() );
    MultiplyAdjoint( alpha, A, B, Scalar(0), C );
}

// Form a dense matrix from a low-rank matrix times a dense matrix
template<typename Scalar>
void MultiplyAdjoint
( Scalar alpha, const LowRank<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyAdjoint (D := F D^H + D)");
    if( A.Width() != B.Width() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( A.Height() != C.Height() )
        throw std::logic_error("The height of A and C are nonconformal.");
    if( B.Height() != C.Width() )
        throw std::logic_error("The width of B and C are nonconformal.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    const int m = A.Height();
    const int n = B.Height();
    const int r = A.Rank();

    // C := alpha (A.U A.V^T) B^H + beta C
    //    = alpha A.U (A.V^T B^H) + beta C
    //
    // W := A.V^T B^H
    // C := alpha A.U W + beta C
    Dense<Scalar> W( r, n );
    blas::Gemm
    ( 'T', 'C', r, n, A.Width(),
      1, A.V.LockedBuffer(), A.V.LDim(),
         B.LockedBuffer(),   B.LDim(),
      0, W.Buffer(),         W.LDim() );
    blas::Gemm
    ( 'N', 'N', m, n, r,
      alpha, A.U.LockedBuffer(), A.U.LDim(),
             W.LockedBuffer(),      W.LDim(),
      beta,  C.Buffer(),            C.LDim() );
}

// Form a dense matrix from the product of two low-rank matrices
template<typename Scalar>
void MultiplyAdjoint
( Scalar alpha, const LowRank<Scalar>& A,
                const LowRank<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyAdjoint (D := F F^H)");
#endif
    C.SetType( GENERAL ); C.Resize( A.Height(), B.Height() );
    MultiplyAdjoint( alpha, A, B, Scalar(0), C );
}

// Update a dense matrix from the product of two low-rank matrices
template<typename Scalar>
void MultiplyAdjoint
( Scalar alpha, const LowRank<Scalar>& A,
                const LowRank<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyAdjoint (D := F F^H + D)");
    if( A.Height() != C.Height() )
        throw std::logic_error("The height of A and C are nonconformal.");
    if( B.Height() != C.Width() )
        throw std::logic_error("The width of B and C are nonconformal.");
#endif
    Dense<Scalar> W( A.Rank(), B.Rank() );
    blas::Gemm
    ( 'C', 'N', A.Rank(), B.Rank(), A.Width(),
      1, A.V.LockedBuffer(), A.V.LDim(), B.V.LockedBuffer(), B.V.LDim(),
      0, W.Buffer(), W.LDim() );
    Conjugate( W );
    Dense<Scalar> X( A.Height(), B.Rank() );
    blas::Gemm
    ( 'N', 'N', A.Height(), B.Rank(), A.Rank(),
      1, A.U.LockedBuffer(), A.U.LDim(), W.LockedBuffer(), W.LDim(),
      0, X.Buffer(), X.LDim() );
    blas::Gemm
    ( 'N', 'C', C.Height(), C.Width(), B.Rank(),
      alpha, X.LockedBuffer(), X.LDim(), B.U.LockedBuffer(), B.U.LDim(),
      beta,  C.Buffer(), C.LDim() );
}

// Low-rank C := alpha A B^H
template<typename Scalar>
void MultiplyAdjoint
( Scalar alpha, const LowRank<Scalar>& A,
                const LowRank<Scalar>& B,
                      LowRank<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyAdjoint (F := F F^H)");
    if( A.Width() != B.Width() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Height();
    const int n = B.Height();
    const int Ar = A.Rank();
    const int Br = B.Rank();
    if( Ar <= Br )
    {
        const int r = Ar;
        C.U.SetType( GENERAL ); C.U.Resize( m, r );
        C.V.SetType( GENERAL ); C.V.Resize( n, r );

        // C.U C.V^T = alpha (A.U A.V^T) (B.U B.V^T)^H
        //           = alpha A.U A.V^T conj(B.V) B.U^H
        //           = A.U (alpha conj(B.U) (B.V^H A.V)))^T
        //           = A.U (alpha conj(B.U) W)^T
        //
        // C.U := A.U
        // W := B.V^H A.V
        // C.V := alpha conj(B.U) W
        Copy( A.U, C.U );
        Dense<Scalar> W( Br, Ar ), cBU;
        blas::Gemm
        ( 'C', 'N', Br, Ar, A.Width(),
          1, B.V.LockedBuffer(), B.V.LDim(),
             A.V.LockedBuffer(), A.V.LDim(),
          0, W.Buffer(),         W.LDim() );
        Conjugate( B.U, cBU );
        blas::Gemm
        ( 'N', 'N', n, Ar, Br,
          alpha, cBU.LockedBuffer(), cBU.LDim(),
                 W.LockedBuffer(),   W.LDim(),
          0,     C.V.Buffer(),       C.V.LDim() );
    }
    else // B.r < A.r
    {
        const int r = Br;
        C.U.SetType( GENERAL ); C.U.Resize( m, r );
        C.V.SetType( GENERAL ); C.V.Resize( n, r );

        // C.U C.V^T := alpha (A.U A.V^T) (B.U B.V^T)^H
        //            = alpha A.U A.V^T conj(B.V) B.U^H
        //            = (alpha A.U A.V^T conj(B.U)) conj(B.U)^T
        //            = (alpha A.U W) conj(B.U)^T
        //
        // W := A.V^T conj(B.U)
        // C.U := alpha A.U W
        // C.V := conj(B.U)
        Dense<Scalar> W( Ar, Br ), cBU;
        Conjugate( B.U, cBU );
        blas::Gemm
        ( 'T', 'N', Ar, Br, B.Height(),
          1, A.V.LockedBuffer(), A.V.LDim(),
             cBU.LockedBuffer(), cBU.LDim(),
          0, W.Buffer(),         W.LDim() );
        blas::Gemm
        ( 'N', 'N', A.Height(), Br, Ar,
          alpha, A.U.LockedBuffer(), A.U.LDim(),
                       W.LockedBuffer(),   W.LDim(),
          0,           C.U.Buffer(),       C.U.LDim() );
        Conjugate( B.U, C.V );
    }
}

// Form a low-rank matrix from a dense matrix times a low-rank matrix
template<typename Scalar>
void MultiplyAdjoint
( Scalar alpha, const Dense<Scalar>& A,
                const LowRank<Scalar>& B,
                      LowRank<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyAdjoint (F := D F^H)");
    if( A.Width() != B.Width() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Height();
    const int n = B.Height();
    const int r = B.Rank();

    C.U.SetType( GENERAL ); C.U.Resize( m, r );
    C.V.SetType( GENERAL ); C.V.Resize( n, r );

    // C.U C.V^T := alpha A conj(B.V) B.U^H
    //                = (alpha A conj(B.V)) B.U^H
    Dense<Scalar> cBV;
    Conjugate( B.V, cBV );
    blas::Gemm
    ( 'N', 'N', m, r, A.Width(),
      alpha, A.LockedBuffer(),   A.LDim(),
             cBV.LockedBuffer(), cBV.LDim(),
      0,     C.U.Buffer(),       C.U.LDim() );
    Conjugate( B.U, C.V );
}

// Form a low-rank matrix from a low-rank matrix times a dense matrix
template<typename Scalar>
void MultiplyAdjoint
( Scalar alpha, const LowRank<Scalar>& A,
                const Dense<Scalar>& B,
                      LowRank<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyAdjoint (F := F D^H)");
    if( A.Width() != B.Width() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Height();
    const int n = B.Height();
    const int r = A.Rank();

    C.U.SetType( GENERAL ); C.U.Resize( m, r );
    C.V.SetType( GENERAL ); C.V.Resize( n, r );

    // C.U C.V^T := alpha A.U A.V^T B^H
    //            = A.U (alpha conj(B) A.V)^T
    //
    // C.U := A.U
    // C.V := alpha conj(B) A.V
    Copy( A.U, C.U );
    Dense<Scalar> cB;
    Conjugate( B, cB );
    blas::Gemm
    ( 'N', 'N', n, r, A.Width(),
      alpha, cB.LockedBuffer(),      cB.LDim(),
             A.V.LockedBuffer(), A.V.LDim(),
      0,     C.V.Buffer(),          C.V.LDim() );
}

template<typename Real>
void MultiplyAdjoint
( int maxRank, Real alpha,
  const Dense<Real>& A,
  const Dense<Real>& B,
        LowRank<Real>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyAdjoint (F := D D^H)");
#endif
    MultiplyTranspose( maxRank, alpha, A, B, C );
}

template<typename Real>
void MultiplyAdjoint
( int maxRank, std::complex<Real> alpha,
  const Dense<std::complex<Real> >& A,
  const Dense<std::complex<Real> >& B,
        LowRank<std::complex<Real> >& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyAdjoint (F := D D^H)");
#endif
    typedef std::complex<Real> Scalar;

    const int m = A.Height();
    const int n = B.Height();
    const int minDim = std::min( m, n );
    const int r = std::min( minDim, maxRank );

    // C.U := alpha A B^H
    MultiplyAdjoint( alpha, A, B, C.U );

    // Get the economic SVD of C.U, C.U = U Sigma V^H, overwriting C.U with U.
    Vector<Real> s( minDim );
    Dense<Scalar> VH( minDim, n );
    const int lwork = lapack::SVDWorkSize( m, n );
    std::vector<Scalar> work( lwork );
    std::vector<Real> rwork( 5*minDim );
    lapack::SVD
    ( 'O', 'S', m, n, C.U.Buffer(), C.U.LDim(),
      s.Buffer(), 0, 1, VH.Buffer(), VH.LDim(),
      &work[0], lwork, &rwork[0] );

    // Truncate the SVD in-place
    C.U.Resize( m, r );
    s.Resize( r );
    VH.Resize( r, n );

    C.V.SetType( GENERAL ); C.V.Resize( n, r );
    // Put (Sigma V^H)^T = (V^H)^T Sigma into C.V
    const int VHLDim = VH.LDim();
    for( int j=0; j<r; ++j )
    {
        const Real sigma = s.Get(j);
        Scalar* RESTRICT VCol = C.V.Buffer(0,j);
        const Scalar* RESTRICT VHRow = VH.LockedBuffer(j,0);
        for( int i=0; i<n; ++i )
            VCol[i] = sigma*VHRow[i*VHLDim];
    }
}

// Dense C := alpha A B^H
template void MultiplyAdjoint
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
                     Dense<float>& C );
template void MultiplyAdjoint
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
                      Dense<double>& C );
template void MultiplyAdjoint
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
                                   Dense<std::complex<float> >& C );
template void MultiplyAdjoint
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
                                    Dense<std::complex<double> >& C );

// Dense C := alpha A B^H + beta C
template void MultiplyAdjoint
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
  float beta,        Dense<float>& C );
template void MultiplyAdjoint
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
  double beta,        Dense<double>& C );
template void MultiplyAdjoint
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
  std::complex<float> beta,        Dense<std::complex<float> >& C );
template void MultiplyAdjoint
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
  std::complex<double> beta,        Dense<std::complex<double> >& C );

// Form a dense matrix from a dense matrix times a low-rank matrix
template void MultiplyAdjoint
( float alpha, const Dense<float>& A,
               const LowRank<float>& B,
                     Dense<float>& C );
template void MultiplyAdjoint
( double alpha, const Dense<double>& A,
                const LowRank<double>& B,
                      Dense<double>& C );
template void MultiplyAdjoint
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const LowRank<std::complex<float> >& B,
                                   Dense<std::complex<float> >& C );
template void MultiplyAdjoint
( std::complex<double> alpha,
  const Dense<std::complex<double> >& A,
  const LowRank<std::complex<double> >& B,
        Dense<std::complex<double> >& C );

// Form a dense matrix from a dense matrix times a low-rank matrix
template void MultiplyAdjoint
( float alpha, const Dense<float>& A,
               const LowRank<float>& B,
  float beta,        Dense<float>& C );
template void MultiplyAdjoint
( double alpha, const Dense<double>& A,
                const LowRank<double>& B,
  double beta,        Dense<double>& C );
template void MultiplyAdjoint
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const LowRank<std::complex<float> >& B,
  std::complex<float> beta,        Dense<std::complex<float> >& C );
template void MultiplyAdjoint
( std::complex<double> alpha,
  const Dense<std::complex<double> >& A,
  const LowRank<std::complex<double> >& B,
  std::complex<double> beta,
        Dense<std::complex<double> >& C );

// Form a dense matrix from a low-rank matrix times a dense matrix
template void MultiplyAdjoint
( float alpha, const LowRank<float>& A,
               const Dense<float>& B,
                     Dense<float>& C );
template void MultiplyAdjoint
( double alpha, const LowRank<double>& A,
                const Dense<double>& B,
                      Dense<double>& C );
template void MultiplyAdjoint
( std::complex<float> alpha, const LowRank<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
                                   Dense<std::complex<float> >& C );
template void MultiplyAdjoint
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const Dense<std::complex<double> >& B,
        Dense<std::complex<double> >& C );

// Form a dense matrix from a low-rank matrix times a dense matrix
template void MultiplyAdjoint
( float alpha, const LowRank<float>& A,
               const Dense<float>& B,
  float beta,        Dense<float>& C );
template void MultiplyAdjoint
( double alpha, const LowRank<double>& A,
                const Dense<double>& B,
  double beta,        Dense<double>& C );
template void MultiplyAdjoint
( std::complex<float> alpha, const LowRank<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
  std::complex<float> beta,        Dense<std::complex<float> >& C );
template void MultiplyAdjoint
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const Dense<std::complex<double> >& B,
  std::complex<double> beta,
        Dense<std::complex<double> >& C );

// Form a dense matrix from the product of two low-rank matrices
template void MultiplyAdjoint
( float alpha, const LowRank<float>& A,
               const LowRank<float>& B,
                     Dense<float>& C );
template void MultiplyAdjoint
( double alpha, const LowRank<double>& A,
                const LowRank<double>& B,
                      Dense<double>& C );
template void MultiplyAdjoint
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const LowRank<std::complex<float> >& B,
        Dense<std::complex<float> >& C );
template void MultiplyAdjoint
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const LowRank<std::complex<double> >& B,
        Dense<std::complex<double> >& C );

// Update a dense matrix from the product of two low-rank matrices
template void MultiplyAdjoint
( float alpha, const LowRank<float>& A,
               const LowRank<float>& B,
  float beta,        Dense<float>& C );
template void MultiplyAdjoint
( double alpha, const LowRank<double>& A,
                const LowRank<double>& B,
  double beta,        Dense<double>& C );
template void MultiplyAdjoint
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const LowRank<std::complex<float> >& B,
  std::complex<float> beta,
        Dense<std::complex<float> >& C );
template void MultiplyAdjoint
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const LowRank<std::complex<double> >& B,
  std::complex<double> beta,
        Dense<std::complex<double> >& C );

// Low-rank C := alpha A B^H
template void MultiplyAdjoint
( float alpha, const LowRank<float>& A,
               const LowRank<float>& B,
                     LowRank<float>& C );
template void MultiplyAdjoint
( double alpha, const LowRank<double>& A,
                const LowRank<double>& B,
                      LowRank<double>& C );
template void MultiplyAdjoint
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const LowRank<std::complex<float> >& B,
        LowRank<std::complex<float> >& C );
template void MultiplyAdjoint
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const LowRank<std::complex<double> >& B,
        LowRank<std::complex<double> >& C );

// Form a low-rank matrix from a dense matrix times a low-rank matrix
template void MultiplyAdjoint
( float alpha, const Dense<float>& A,
               const LowRank<float>& B,
                     LowRank<float>& C );
template void MultiplyAdjoint
( double alpha, const Dense<double>& A,
                const LowRank<double>& B,
                      LowRank<double>& C );
template void MultiplyAdjoint
( std::complex<float> alpha,
  const Dense<std::complex<float> >& A,
  const LowRank<std::complex<float> >& B,
        LowRank<std::complex<float> >& C );
template void MultiplyAdjoint
( std::complex<double> alpha,
  const Dense<std::complex<double> >& A,
  const LowRank<std::complex<double> >& B,
        LowRank<std::complex<double> >& C );

// Form a low-rank matrix from a low-rank matrix times a dense matrix
template void MultiplyAdjoint
( float alpha, const LowRank<float>& A,
               const Dense<float>& B,
                     LowRank<float>& C );
template void MultiplyAdjoint
( double alpha, const LowRank<double>& A,
                const Dense<double>& B,
                      LowRank<double>& C );
template void MultiplyAdjoint
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const Dense<std::complex<float> >& B,
        LowRank<std::complex<float> >& C );
template void MultiplyAdjoint
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const Dense<std::complex<double> >& B,
        LowRank<std::complex<double> >& C );

} // namespace hmat_tools
} // namespace hifde3d

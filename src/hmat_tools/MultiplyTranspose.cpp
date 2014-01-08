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

// Dense C := alpha A B^T
template<typename Scalar>
void MultiplyTranspose
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyTranspose (D := D D^T)");
#endif
    C.SetType( GENERAL );
    C.Resize( A.Height(), B.Height() );
    MultiplyTranspose( alpha, A, B, Scalar(0), C );
}

// Dense C := alpha A B^T + beta C
template<typename Scalar>
void MultiplyTranspose
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyTranspose (D := D D^T + D)");
    if( A.Width() != B.Width() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( A.Height() != C.Height() )
        throw std::logic_error("The height of A and C are nonconformal.");
    if( B.Height() != C.Width() )
        throw std::logic_error("The width of B and C are nonconformal.");
    if( B.Symmetric() || A.Symmetric() )
        throw std::logic_error("BLAS does not support syms times trans");
#endif
    blas::Gemm
    ( 'N', 'T', C.Height(), C.Width(), A.Width(),
      alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(),
      beta, C.Buffer(), C.LDim() );
}

// Form a dense matrix from a dense matrix times a low-rank matrix
template<typename Scalar>
void MultiplyTranspose
( Scalar alpha, const Dense<Scalar>& A,
                const LowRank<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyTranspose (D := D F^T + D)");
    if( A.Width() != B.Width() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( A.Height() != C.Height() )
        throw std::logic_error("The height of A and C are nonconformal.");
    if( B.Height() != C.Width() )
        throw std::logic_error("The width of B and C are nonconformal.");
    if( C.Symmetric() )
        throw std::logic_error("Update is probably not symmetric.");
#endif
    // W := A B.V
    Dense<Scalar> W( A.Height(), B.Rank() );
    blas::Gemm
    ( 'N', 'N', A.Height(), B.Rank(), A.Width(),
      1, A.LockedBuffer(), A.LDim(), B.V.LockedBuffer(), B.V.LDim(),
      0, W.Buffer(), W.LDim() );
    // C := alpha W B.U^T + beta C
    blas::Gemm
    ( 'N', 'T', C.Height(), C.Width(), B.Rank(),
      alpha, W.LockedBuffer(), W.LDim(), B.U.LockedBuffer(), B.U.LDim(),
      beta,  C.Buffer(), C.LDim() );
}

// Form a dense matrix from a low-rank matrix times a dense matrix
template<typename Scalar>
void MultiplyTranspose
( Scalar alpha, const LowRank<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyTranspose (D := F D^T + D)");
    if( A.Width() != B.Width() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( A.Height() != C.Height() )
        throw std::logic_error("The height of A and C are nonconformal.");
    if( B.Height() != C.Width() )
        throw std::logic_error("The width of B and C are nonconformal.");
    if( C.Symmetric() )
        throw std::logic_error("Update is probably not symmetric.");
#endif
    const int m = A.Height();
    const int n = B.Height();
    const int r = A.Rank();

    // C := alpha (A.U A.V^T) B^T + beta C
    //    = alpha A.U (A.V^T B^T) + beta C
    //
    // W := A.V^T B^T
    // C := alpha A.U W + beta C
    Dense<Scalar> W( r, n );
    blas::Gemm
    ( 'T', 'T', r, n, A.Width(),
      1, A.U.LockedBuffer(), A.V.LDim(), B.LockedBuffer(), B.LDim(),
      0, W.Buffer(), W.LDim() );
    blas::Gemm
    ( 'N', 'N', m, n, r,
      alpha, A.U.LockedBuffer(), A.U.LDim(), W.LockedBuffer(), W.LDim(),
      beta,  C.Buffer(), C.LDim() );
}

// Update a dense matrix from the product of two low-rank matrices
template<typename Scalar>
void MultiplyTranspose
( Scalar alpha, const LowRank<Scalar>& A,
                const LowRank<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyTranspose (D := F F^T + D)");
    if( A.Height() != C.Height() )
        throw std::logic_error("The height of A and C are nonconformal.");
    if( B.Height() != C.Width() )
        throw std::logic_error("The width of B and C are nonconformal.");
#endif
    Dense<Scalar> W( A.Rank(), B.Rank() );
    blas::Gemm
    ( 'T', 'N', A.Rank(), B.Rank(), A.Width(),
      1, A.V.LockedBuffer(), A.V.LDim(), B.V.LockedBuffer(), B.V.LDim(),
      0, W.Buffer(), W.LDim() );
    Dense<Scalar> X( A.Height(), B.Rank() );
    blas::Gemm
    ( 'N', 'N', A.Height(), B.Rank(), A.Rank(),
      1, A.U.LockedBuffer(), A.U.LDim(), W.LockedBuffer(), W.LDim(),
      0, X.Buffer(), X.LDim() );
    blas::Gemm
    ( 'N', 'T', C.Height(), C.Width(), B.Rank(),
      alpha, X.LockedBuffer(), X.LDim(), B.U.LockedBuffer(), B.U.LDim(),
      beta,  C.Buffer(), C.LDim() );
}

// Low-rank C := alpha A B^T
template<typename Scalar>
void MultiplyTranspose
( Scalar alpha, const LowRank<Scalar>& A,
                const LowRank<Scalar>& B,
                      LowRank<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyTranspose (F := F^T F)");
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

        // C.U C.V^T := alpha (A.U A.V^T) (B.U B.V^T)^T
        //            = alpha A.U A.V^T B.V B.U^T
        //            = A.U (alpha A.V^T B.V B.U^T)
        //            = A.U (alpha B.U (B.V^T A.V))^T
        //
        // C.U := A.U
        // W := B.V^T A.V
        // C.V := alpha B.U W
        Copy( A.U, C.U );
        Dense<Scalar> W( Br, Ar );
        blas::Gemm
        ( 'T', 'N', Br, Ar, B.Width(),
          1, B.V.LockedBuffer(), B.V.LDim(),
             A.V.LockedBuffer(), A.V.LDim(),
          0, W.Buffer(),         W.LDim() );
        blas::Gemm
        ( 'N', 'N', n, Ar, Br,
          alpha, B.U.LockedBuffer(), B.U.LDim(),
                 W.LockedBuffer(),   W.LDim(),
          0,     C.V.Buffer(),       C.V.LDim() );
    }
    else // B.r < A.r
    {
        const int r = Br;
        C.U.SetType( GENERAL ); C.U.Resize( m, r );
        C.V.SetType( GENERAL ); C.V.Resize( n, r );

        // C.U C.V^T := alpha (A.U A.V^T) (B.U B.V^T)^T
        //            = alpha A.U A.V^T B.V B.U^T
        //            = (alpha A.U (A.V^T B.V)) B.U^T
        //
        // W := A.V^T B.V
        // C.U := alpha A.U W
        // C.V := B.U
        Dense<Scalar> W( Ar, Br );
        blas::Gemm
        ( 'T', 'N', Ar, Br, A.Width(),
          1, A.V.LockedBuffer(), A.V.LDim(),
             B.V.LockedBuffer(), B.V.LDim(),
          0, W.Buffer(),         W.LDim() );
        blas::Gemm
        ( 'N', 'N', m, Br, Ar,
          alpha, A.U.LockedBuffer(), A.U.LDim(),
                 W.LockedBuffer(),   W.LDim(),
          0,     C.U.Buffer(),       C.U.LDim() );
        Copy( B.U, C.V );
    }
}

// Form a low-rank matrix from a dense matrix times a low-rank matrix
template<typename Scalar>
void MultiplyTranspose
( Scalar alpha, const Dense<Scalar>& A,
                const LowRank<Scalar>& B,
                      LowRank<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyTranspose (F := D F^T)");
    if( A.Width() != B.Width() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Height();
    const int n = B.Height();
    const int r = B.Rank();

    C.U.SetType( GENERAL ); C.U.Resize( m, r );
    C.V.SetType( GENERAL ); C.V.Resize( n, r );

    blas::Gemm
    ( 'N', 'N', m, r, A.Width(),
      alpha, A.LockedBuffer(),   A.LDim(),
             B.V.LockedBuffer(), B.V.LDim(),
      0,     C.U.Buffer(),       C.U.LDim() );

    // Form C.V := B.U
    Copy( B.U, C.V );
}

// Form a low-rank matrix from a low-rank matrix times a dense matrix
template<typename Scalar>
void MultiplyTranspose
( Scalar alpha, const LowRank<Scalar>& A,
                const Dense<Scalar>& B,
                      LowRank<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyTranspose (F := F D^T)");
    if( A.Width() != B.Width() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Height();
    const int n = B.Height();
    const int r = A.Rank();

    C.U.SetType( GENERAL ); C.U.Resize( m, r );
    C.V.SetType( GENERAL ); C.V.Resize( n, r );

    // C.U C.V^T := alpha (A.U A.V^T) B^T
    //            = alpha A.U A.V^T B^T
    //            = A.U (alpha B A.V)^T
    //
    // C.U := A.U
    // C.V := alpha B A.V
    Copy( A.U, C.U );
    blas::Gemm
    ( 'N', 'N', n, r, A.Width(),
      alpha, B.LockedBuffer(),   B.LDim(),
             A.V.LockedBuffer(), A.V.LDim(),
      0,     C.V.Buffer(),       C.V.LDim() );
}

// Form a low-rank matrix from the product of two dense matrices
template<typename Real>
void MultiplyTranspose
( int maxRank, Real alpha,
  const Dense<Real>& A,
  const Dense<Real>& B,
        LowRank<Real>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyTranspose (F := D D^T)");
#endif
    const int m = A.Height();
    const int n = B.Height();
    const int minDim = std::min( m, n );
    const int r = std::min( minDim, maxRank );

    // C.U := alpha A B^T
    MultiplyTranspose( alpha, A, B, C.U );

    // Get the economic SVD of C.U, C.U = U Sigma V^T, overwriting C.U with U.
    Vector<Real> s( minDim );
    Dense<Real> VT( minDim, n );
    const int lwork = lapack::SVDWorkSize( m, n );
    std::vector<Real> work( lwork );
    lapack::SVD
    ( 'O', 'S', m, n, C.U.Buffer(), C.U.LDim(),
      s.Buffer(), 0, 1, VT.Buffer(), VT.LDim(),
      &work[0], lwork );

    // Truncate the SVD in-place
    C.U.Resize( m, r );
    s.Resize( r );
    VT.Resize( r, n );

    // Put (Sigma V^T)^T = V Sigma into C.V
    C.V.SetType( GENERAL ); C.V.Resize( n, r );
    const int VTLDim = VT.LDim();
    for( int j=0; j<r; ++j )
    {
        const Real sigma = s.Get(j);
        Real* RESTRICT VCol = C.V.Buffer(0,j);
        const Real* RESTRICT VTRow = VT.LockedBuffer(j,0);
        for( int i=0; i<n; ++i )
            VCol[i] = sigma*VTRow[i*VTLDim];
    }
}

// Form a low-rank matrix from the product of two dense matrices
template<typename Real>
void MultiplyTranspose
( int maxRank, std::complex<Real> alpha,
  const Dense<std::complex<Real> >& A,
  const Dense<std::complex<Real> >& B,
        LowRank<std::complex<Real> >& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::MultiplyTranspose (F := D D^T)");
#endif
    typedef std::complex<Real> Scalar;

    const int m = A.Height();
    const int n = B.Height();
    const int minDim = std::min( m, n );
    const int r = std::min( minDim, maxRank );

    // C.U := alpha A B^T
    MultiplyTranspose( alpha, A, B, C.U );

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

// Dense C := alpha A B^T
template void MultiplyTranspose
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
                     Dense<float>& C );
template void MultiplyTranspose
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
                      Dense<double>& C );
template void MultiplyTranspose
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
                                   Dense<std::complex<float> >& C );
template void MultiplyTranspose
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
                                    Dense<std::complex<double> >& C );

// Dense C := alpha A B^T + beta C
template void MultiplyTranspose
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
  float beta,        Dense<float>& C );
template void MultiplyTranspose
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
  double beta,        Dense<double>& C );
template void MultiplyTranspose
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
  std::complex<float> beta,        Dense<std::complex<float> >& C );
template void MultiplyTranspose
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
  std::complex<double> beta,        Dense<std::complex<double> >& C );

// Form a dense matrix from a dense matrix times a low-rank matrix
template void MultiplyTranspose
( float alpha, const Dense<float>& A,
               const LowRank<float>& B,
  float beta,        Dense<float>& C );
template void MultiplyTranspose
( double alpha, const Dense<double>& A,
                const LowRank<double>& B,
  double beta,        Dense<double>& C );
template void MultiplyTranspose
( std::complex<float> alpha,
  const Dense<std::complex<float> >& A,
  const LowRank<std::complex<float> >& B,
  std::complex<float> beta,
        Dense<std::complex<float> >& C );
template void MultiplyTranspose
( std::complex<double> alpha,
  const Dense<std::complex<double> >& A,
  const LowRank<std::complex<double> >& B,
  std::complex<double> beta,
        Dense<std::complex<double> >& C );

// Form a dense matrix from a low-rank matrix times a dense matrix
template void MultiplyTranspose
( float alpha, const LowRank<float>& A,
               const Dense<float>& B,
  float beta,        Dense<float>& C );
template void MultiplyTranspose
( double alpha, const LowRank<double>& A,
                const Dense<double>& B,
  double beta,        Dense<double>& C );
template void MultiplyTranspose
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const Dense<std::complex<float> >& B,
  std::complex<float> beta,
        Dense<std::complex<float> >& C );
template void MultiplyTranspose
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const Dense<std::complex<double> >& B,
  std::complex<double> beta,
        Dense<std::complex<double> >& C );

// Update a dense matrix as the product of two low-rank matrices
template void MultiplyTranspose
( float alpha, const LowRank<float>& A,
               const LowRank<float>& B,
  float beta,        Dense<float>& C );
template void MultiplyTranspose
( double alpha, const LowRank<double>& A,
                const LowRank<double>& B,
  double beta,        Dense<double>& C );
template void MultiplyTranspose
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const LowRank<std::complex<float> >& B,
  std::complex<float> beta,
        Dense<std::complex<float> >& C );
template void MultiplyTranspose
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const LowRank<std::complex<double> >& B,
  std::complex<double> beta,
        Dense<std::complex<double> >& C );

// Low-rank C := alpha A B^T
template void MultiplyTranspose
( float alpha, const LowRank<float>& A,
               const LowRank<float>& B,
                     LowRank<float>& C );
template void MultiplyTranspose
( double alpha, const LowRank<double>& A,
                const LowRank<double>& B,
                      LowRank<double>& C );
template void MultiplyTranspose
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const LowRank<std::complex<float> >& B,
        LowRank<std::complex<float> >& C );
template void MultiplyTranspose
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const LowRank<std::complex<double> >& B,
        LowRank<std::complex<double> >& C );

// Form a low-rank matrix from a dense matrix times a low-rank matrix
template void MultiplyTranspose
( float alpha, const Dense<float>& A,
               const LowRank<float>& B,
                     LowRank<float>& C );
template void MultiplyTranspose
( double alpha, const Dense<double>& A,
                const LowRank<double>& B,
                      LowRank<double>& C );
template void MultiplyTranspose
( std::complex<float> alpha,
  const Dense<std::complex<float> >& A,
  const LowRank<std::complex<float> >& B,
        LowRank<std::complex<float> >& C );
template void MultiplyTranspose
( std::complex<double> alpha,
  const Dense<std::complex<double> >& A,
  const LowRank<std::complex<double> >& B,
        LowRank<std::complex<double> >& C );

// Form a low-rank matrix from a low-rank matrix times a dense matrix
template void MultiplyTranspose
( float alpha, const LowRank<float>& A,
               const Dense<float>& B,
                     LowRank<float>& C );
template void MultiplyTranspose
( double alpha, const LowRank<double>& A,
                const Dense<double>& B,
                      LowRank<double>& C );
template void MultiplyTranspose
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const Dense<std::complex<float> >& B,
        LowRank<std::complex<float> >& C );
template void MultiplyTranspose
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const Dense<std::complex<double> >& B,
        LowRank<std::complex<double> >& C );

// Generate a low-rank matrix from the product of two dense matrices
template void MultiplyTranspose
( int maxRank, float alpha,
  const Dense<float>& A,
  const Dense<float>& B,
        LowRank<float>& C );
template void MultiplyTranspose
( int maxRank, double alpha,
  const Dense<double>& A,
  const Dense<double>& B,
        LowRank<double>& C );
template void MultiplyTranspose
( int maxRank, std::complex<float> alpha,
  const Dense<std::complex<float> >& A,
  const Dense<std::complex<float> >& B,
        LowRank<std::complex<float> >& C );
template void MultiplyTranspose
( int maxRank, std::complex<double> alpha,
  const Dense<std::complex<double> >& A,
  const Dense<std::complex<double> >& B,
        LowRank<std::complex<double> >& C );

} // namespace hmat_tools
} // namespace hifde3d

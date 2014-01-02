/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "dmhm.hpp"

namespace dmhm {
namespace hmat_tools {

// Dense C := alpha A^T B
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::TransposeMultiply (D := D^T D)");
#endif
    C.SetType( GENERAL );
    C.Resize( A.Width(), B.Width() );
    TransposeMultiply( alpha, A, B, Scalar(0), C );
}

// Dense C := alpha A^T B + beta C
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::TransposeMultiply (D := D^T D + D)");
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( B.Symmetric() )
        throw std::logic_error("BLAS does not support symm times trans");
#endif
    if( A.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', C.Height(), C.Width(),
          alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(),
          beta, C.Buffer(), C.LDim() );
    }
    else
    {
        blas::Gemm
        ( 'T', 'N', C.Height(), C.Width(), A.Height(),
          alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(),
          beta, C.Buffer(), C.LDim() );
    }
}

// Form a dense matrix from a dense matrix times a low-rank matrix
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const LowRank<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::TransposeMultiply (D := D^T F + D)");
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Symmetric() )
        throw std::logic_error("Update is probably not symmetric.");
#endif
    // W := A^T B.U
    Dense<Scalar> W( A.Width(), B.Rank() );
    if( A.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', A.Width(), B.Rank(),
          1, A.LockedBuffer(), A.LDim(), B.U.LockedBuffer(), B.U.LDim(),
          0, W.Buffer(), W.LDim() );
    }
    else
    {
        blas::Gemm
        ( 'T', 'N', A.Width(), B.Rank(), A.Height(),
          1, A.LockedBuffer(), A.LDim(), B.U.LockedBuffer(), B.U.LDim(),
          0, W.Buffer(), W.LDim() );
    }
    // C := alpha W B.V^[T,H] + beta C
    const char option = 'T';
    blas::Gemm
    ( 'N', option, C.Height(), C.Width(), B.Rank(),
      alpha, W.LockedBuffer(), W.LDim(), B.V.LockedBuffer(), B.V.LDim(),
      beta,  C.Buffer(), C.LDim() );
}

// Low-rank C := alpha A^T B
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const LowRank<Scalar>& A,
                const LowRank<Scalar>& B,
                      LowRank<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::TransposeMultiply (F := F^T F)");
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Width();
    const int n = B.Width();
    const int Ar = A.Rank();
    const int Br = B.Rank();

    if( Ar <= Br )
    {
        const int r = Ar;
        C.U.SetType( GENERAL ); C.U.Resize( m, r );
        C.V.SetType( GENERAL ); C.V.Resize( n, r );

        // C.U C.V^T := alpha (A.U A.V^T)^T (B.U B.V^T)
        //            = alpha A.V A.U^T B.U B.V^T
        //            = A.V (alpha A.U^T B.U B.V^T)
        //            = A.V (alpha B.V (B.U^T A.U))^T
        //
        // C.U := A.V
        // W := B.U^T A.U
        // C.V := alpha B.V W
        Copy( A.V, C.U );
        Dense<Scalar> W( Br, Ar );
        blas::Gemm
        ( 'T', 'N', Br, Ar, B.Height(),
          1, B.U.LockedBuffer(), B.U.LDim(),
             A.U.LockedBuffer(), A.U.LDim(),
          0, W.Buffer(),         W.LDim() );
        blas::Gemm
        ( 'N', 'N', n, Ar, Br,
          alpha, B.V.LockedBuffer(), B.V.LDim(),
                 W.LockedBuffer(),   W.LDim(),
          0,     C.V.Buffer(),       C.V.LDim() );
    }
    else // B.r < A.r
    {
        const int r = Br;
        C.U.SetType( GENERAL ); C.U.Resize( m, r );
        C.V.SetType( GENERAL ); C.V.Resize( n, r );

        // C.U C.V^T := alpha (A.U A.V^T)^T (B.U B.V^T)
        //            = alpha A.V A.U^T B.U B.V^T
        //            = (alpha A.V (A.U^T B.U)) B.V^T
        //
        // W := A.U^T B.U
        // C.U := alpha A.V W
        // C.V := B.V
        Dense<Scalar> W( Ar, Br );
        blas::Gemm
        ( 'T', 'N', Ar, Br, A.Height(),
          1, A.U.LockedBuffer(), A.U.LDim(),
             B.U.LockedBuffer(), B.U.LDim(),
          0, W.Buffer(),         W.LDim() );
        blas::Gemm
        ( 'N', 'N', m, Br, Ar,
          alpha, A.V.LockedBuffer(), A.V.LDim(),
                 W.LockedBuffer(),   W.LDim(),
          0,     C.U.Buffer(),       C.U.LDim() );
        Copy( B.V, C.V );
    }
}

// Form a low-rank matrix from a dense matrix times a low-rank matrix
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const LowRank<Scalar>& B,
                      LowRank<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::TransposeMultiply (F := D^T F)");
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Width();
    const int n = B.Width();
    const int r = B.Rank();

    C.U.SetType( GENERAL ); C.U.Resize( m, r );
    C.V.SetType( GENERAL ); C.V.Resize( n, r );

    // Form C.U := A B.U
    if( A.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', m, r,
          alpha, A.LockedBuffer(),   A.LDim(),
                 B.U.LockedBuffer(), B.U.LDim(),
          0,     C.U.Buffer(),       C.U.LDim() );
    }
    else
    {
        blas::Gemm
        ( 'T', 'N', m, r, A.Height(),
          alpha, A.LockedBuffer(),   A.LDim(),
                 B.U.LockedBuffer(), B.U.LDim(),
          0,     C.U.Buffer(),       C.U.LDim() );
    }

    // Form C.V := B.V
    Copy( B.V, C.V );
}

// Form a low-rank matrix from a low-rank matrix times a dense matrix
template<typename Scalar>
void TransposeMultiply
( Scalar alpha, const LowRank<Scalar>& A,
                const Dense<Scalar>& B,
                      LowRank<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::TransposeMultiply (F := F^T D)");
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Width();
    const int n = B.Width();
    const int r = A.Rank();

    C.U.SetType( GENERAL ); C.U.Resize( m, r );
    C.V.SetType( GENERAL ); C.V.Resize( n, r );

    if( B.Symmetric() )
    {
        // C.U C.V^T := alpha (A.U A.V^T)^T B
        //            = alpha A.V A.U^T B
        //            = A.V (alpha A.U^T B)
        //            = A.V (alpha B A.U)^T
        //
        // C.U := A.V
        // C.V := alpha B A.U
        Copy( A.V, C.U );
        blas::Symm
        ( 'L', 'L', A.Height(), r,
          alpha, B.LockedBuffer(),   B.LDim(),
                 A.U.LockedBuffer(), A.U.LDim(),
          0,     C.V.Buffer(),       C.V.LDim() );
    }
    else
    {
        // C.U C.V^T := alpha (A.U A.V^T)^T B
        //            = alpha A.V A.U^T B
        //            = A.V (alpha B^T A.U)^T
        //
        // C.U := A.V
        // C.V := alpha B^T A.U
        Copy( A.V, C.U );
        blas::Gemm
        ( 'T', 'N', n, r, A.Height(),
          alpha, B.LockedBuffer(),   B.LDim(),
                 A.U.LockedBuffer(), A.U.LDim(),
          0,     C.V.Buffer(),       C.V.LDim() );
    }
}

// Form a low-rank matrix from the product of two dense matrices
template<typename Real>
void TransposeMultiply
( int maxRank, Real alpha,
  const Dense<Real>& A,
  const Dense<Real>& B,
        LowRank<Real>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::TransposeMultiply (F := D^T D)");
#endif
    const int m = A.Width();
    const int n = B.Width();
    const int minDim = std::min( m, n );
    const int r = std::min( minDim, maxRank );

    // C.U := alpha A^T B
    TransposeMultiply( alpha, A, B, C.U );

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
void TransposeMultiply
( int maxRank, std::complex<Real> alpha,
  const Dense<std::complex<Real> >& A,
  const Dense<std::complex<Real> >& B,
        LowRank<std::complex<Real> >& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::TransposeMultiply (F := D^T D)");
#endif
    typedef std::complex<Real> Scalar;

    const int m = A.Width();
    const int n = B.Width();
    const int minDim = std::min( m, n );
    const int r = std::min( minDim, maxRank );

    // C.U := alpha A^T B
    TransposeMultiply( alpha, A, B, C.U );

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

// Dense C := alpha A^T B
template void TransposeMultiply
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
                     Dense<float>& C );
template void TransposeMultiply
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
                      Dense<double>& C );
template void TransposeMultiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
                                   Dense<std::complex<float> >& C );
template void TransposeMultiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
                                    Dense<std::complex<double> >& C );

// Dense C := alpha A^T B + beta C
template void TransposeMultiply
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
  float beta,        Dense<float>& C );
template void TransposeMultiply
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
  double beta,        Dense<double>& C );
template void TransposeMultiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
  std::complex<float> beta,        Dense<std::complex<float> >& C );
template void TransposeMultiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
  std::complex<double> beta,        Dense<std::complex<double> >& C );

// Form a low-rank matrix from a low-rank matrix times a dense matrix
template void TransposeMultiply
( float alpha, const LowRank<float>& A,
               const Dense<float>& B,
                     LowRank<float>& C );
template void TransposeMultiply
( double alpha, const LowRank<double>& A,
                const Dense<double>& B,
                      LowRank<double>& C );
template void TransposeMultiply
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const Dense<std::complex<float> >& B,
        LowRank<std::complex<float> >& C );
template void TransposeMultiply
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const Dense<std::complex<double> >& B,
        LowRank<std::complex<double> >& C );

// Generate a low-rank matrix from the product of two dense matrices
template void TransposeMultiply
( int maxRank, float alpha,
  const Dense<float>& A,
  const Dense<float>& B,
        LowRank<float>& C );
template void TransposeMultiply
( int maxRank, double alpha,
  const Dense<double>& A,
  const Dense<double>& B,
        LowRank<double>& C );
template void TransposeMultiply
( int maxRank, std::complex<float> alpha,
  const Dense<std::complex<float> >& A,
  const Dense<std::complex<float> >& B,
        LowRank<std::complex<float> >& C );
template void TransposeMultiply
( int maxRank, std::complex<double> alpha,
  const Dense<std::complex<double> >& A,
  const Dense<std::complex<double> >& B,
        LowRank<std::complex<double> >& C );

} // namespace hmat_tools
} // namespace dmhm

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

// Dense C := alpha A^H B
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::AdjointMultiply (D := D^H D)");
#endif
    C.SetType( GENERAL );
    C.Resize( A.Width(), B.Width() );
    AdjointMultiply( alpha, A, B, Scalar(0), C );
}

// Dense C := alpha A^H B + beta C
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry
    ("hmat_tools::AdjointMultiply (D := D^H D + D)");
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( B.Symmetric() )
        throw std::logic_error("BLAS does not support symm times trans.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    if( A.Symmetric() )
    {
        const int m = A.Height();
        const int n = B.Width();
        if( m <= 2*n )
        {
            // C := alpha A^H B + beta C
            //    = alpha conj(A) B + beta C
            //
            // AConj := conj(A)
            // C := alpha AConj B + beta C

            Dense<Scalar> AConj(m,m);
            Conjugate( A, AConj );
            blas::Symm
            ( 'L', 'L', C.Height(), C.Width(),
              alpha, AConj.LockedBuffer(), AConj.LDim(),
                     B.LockedBuffer(), B.LDim(),
              beta,  C.Buffer(), C.LDim() );
        }
        else
        {
            // C := alpha A^H B + beta C
            //    = alpha conj(A) B + beta C
            //    = conj(conj(alpha) A conj(B) + conj(beta) conj(C))
            //
            // BConj := conj(B)
            // CConj := conj(C)
            // C := conj(alpha) A BConj + conj(beta) CConj
            // C := conj(C)

            Dense<Scalar> BConj(m,n);
            Conjugate( B, BConj );
            Conjugate( C );
            blas::Symm
            ( 'L', 'L', C.Height(), C.Width(),
              Conj(alpha), A.LockedBuffer(), A.LDim(),
                           BConj.LockedBuffer(), BConj.LDim(),
              Conj(beta),  C.Buffer(), C.LDim() );
            Conjugate( C );
        }
    }
    else
    {
        blas::Gemm
        ( 'C', 'N', C.Height(), C.Width(), A.Height(),
          alpha, A.LockedBuffer(), A.LDim(), B.LockedBuffer(), B.LDim(),
          beta, C.Buffer(), C.LDim() );
    }
}

// TODO: version of above routine that allows for temporary in-place conj of B

// Form a dense matrix from a dense matrix times a low-rank matrix
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const LowRank<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry
    ("hmat_tools::AdjointMultiply (D := D^H F + D)");
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    const int m = C.Height();
    const int n = C.Width();
    const int r = B.Rank();

    if( A.Symmetric() )
    {
        // C := alpha A^H (B.U B.V^[T,H]) + beta C
        //    = alpha (conj(A) B.U) B.V^[T,H] + beta C
        //    = alpha conj(A conj(B.U)) B.V^[T,H] + beta C
        //
        // BUConj := conj(B.U)
        // W := A BUConj
        // W := conj(W)
        // C := alpha W B.V^[T,H] + beta C
        Dense<Scalar> BUConj;
        Conjugate( B.U, BUConj );
        Dense<Scalar> W( B.Height(), r );
        blas::Symm
        ( 'L', 'L', B.Height(), r,
          1, A.LockedBuffer(),      A.LDim(),
             BUConj.LockedBuffer(), BUConj.LDim(),
          0, W.Buffer(),            W.LDim() );
        Conjugate( W );
        const char option = 'T';
        blas::Gemm
        ( 'N', option, m, n, r,
          alpha, W.LockedBuffer(),   W.LDim(),
                 B.V.LockedBuffer(), B.V.LDim(),
          beta,  C.Buffer(),         C.LDim() );
    }
    else
    {
        // C := alpha (A^H B.U) B.V^[T,H] + beta C
        //
        // W := A^H B.U
        // C := alpha W B.V^[T,H] + beta C
        Dense<Scalar> W( m, r );
        blas::Gemm
        ( 'C', 'N', m, r, B.Height(),
          1, A.LockedBuffer(),   A.LDim(),
             B.U.LockedBuffer(), B.U.LDim(),
          0, W.Buffer(),         W.LDim() );
        const char option = 'T';
        blas::Gemm
        ( 'N', option, m, n, r,
          alpha, W.LockedBuffer(),   W.LDim(),
                 B.V.LockedBuffer(), B.V.LDim(),
          beta,  C.Buffer(),         C.LDim() );
    }
}

// Form a dense matrix from a low-rank matrix times a dense matrix
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const LowRank<Scalar>& A,
                const Dense<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::AdjointMultiply (D := F^H D + D)");
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
    if( C.Symmetric() )
        throw std::logic_error("Update will probably not be symmetric.");
#endif
    const int m = A.Width();
    const int n = B.Width();
    const int r = A.Rank();

    if( B.Symmetric() )
    {
        // C := alpha (A.U A.V^T)^H B + beta C
        //    = alpha conj(A.V) (A.U^H B) + beta C
        //    = alpha conj(A.V) (B conj(A.U))^T + beta C
        //
        // AUConj := conj(A.U)
        // W := B AUConj
        // AVConj := conj(A.V)
        // C := alpha AVConj W^T + beta C
        Dense<Scalar> AUConj;
        Conjugate( A.U, AUConj );
        Dense<Scalar> W( A.Height(), r );
        blas::Symm
        ( 'L', 'L', A.Height(), r,
          1, B.LockedBuffer(),      B.LDim(),
             AUConj.LockedBuffer(), AUConj.LDim(),
          0, W.Buffer(),            W.LDim() );
        Dense<Scalar> AVConj;
        Conjugate( A.V, AVConj );
        blas::Gemm
        ( 'N', 'T', m, A.Height(), r,
          alpha, AVConj.LockedBuffer(), AVConj.LDim(),
                 W.LockedBuffer(),      W.LDim(),
          beta, C.Buffer(),             C.LDim() );
    }
    else
    {
        // C := alpha (A.U A.V^T)^H B + beta C
        //    = alpha conj(A.V) (A.U^H B) + beta C
        //
        // W := A.U^H B
        // AVConj := conj(A.V)
        // C := alpha AVConj W + beta C
        Dense<Scalar> W( r, n );
        blas::Gemm
        ( 'C', 'N', r, n, A.Height(),
          1, A.U.LockedBuffer(), A.U.LDim(),
             B.LockedBuffer(),   B.LDim(),
          0, W.Buffer(),         W.LDim() );
        Dense<Scalar> AVConj;
        Conjugate( A.V, AVConj );
        blas::Gemm
        ( 'N', 'N', m, n, r,
          alpha, AVConj.LockedBuffer(), AVConj.LDim(),
                 W.LockedBuffer(),      W.LDim(),
          beta,  C.Buffer(),            C.LDim() );
    }
}

// Update a dense matrix from the product of two low-rank matrices
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const LowRank<Scalar>& A,
                const LowRank<Scalar>& B,
  Scalar beta,        Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::AdjointMultiply (D := F^H F + D)");
#endif
    Dense<Scalar> W( A.Rank(), B.Rank() );
    blas::Gemm
    ( 'C', 'N', A.Rank(), B.Rank(), A.Height(),
      1, A.U.LockedBuffer(), A.U.LDim(), B.U.LockedBuffer(), B.U.LDim(),
      0, W.Buffer(), W.LDim() );
    Conjugate( W );
    Dense<Scalar> X( A.Width(), B.Rank() );
    blas::Gemm
    ( 'N', 'N', A.Width(), B.Rank(), A.Rank(),
      1, A.V.LockedBuffer(), A.V.LDim(), W.LockedBuffer(), W.LDim(),
      0, X.Buffer(), X.LDim() );
    Conjugate( X );
    blas::Gemm
    ( 'N', 'T', C.Height(), C.Width(), B.Rank(),
      alpha, X.LockedBuffer(), X.LDim(), B.V.LockedBuffer(), B.V.LDim(),
      beta,  C.Buffer(), C.LDim() );
}

// Low-rank C := alpha A^H B
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const LowRank<Scalar>& A,
                const LowRank<Scalar>& B,
                      LowRank<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::AdjointMultiply (F := F^H F)");
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

        // C.U C.V^T = alpha (A.U A.V^T)^H (B.U B.V^T)
        //           = alpha conj(A.V) A.U^H B.U B.V^T
        //           = conj(A.V) (alpha B.V (B.U^T conj(A.U)))^T
        //           = conj(A.V) (alpha B.V (A.U^H B.U)^T)^T
        //           = conj(A.V) (alpha B.V W^T)^T
        //
        // C.U := conj(A.V)
        // W := A.U^H B.U
        // C.V := alpha B.V W^T
        Conjugate( A.V, C.U );
        Dense<Scalar> W( Ar, Br );
        blas::Gemm
        ( 'C', 'N', Ar, Br, A.Height(),
          1, A.U.LockedBuffer(), A.U.LDim(),
             B.U.LockedBuffer(), B.U.LDim(),
          0, W.Buffer(),         W.LDim() );
        blas::Gemm
        ( 'N', 'T', n, Ar, Br,
          alpha, B.V.LockedBuffer(), B.V.LDim(),
                 W.LockedBuffer(),   W.LDim(),
          0,     C.V.Buffer(),       C.V.LDim() );
    }
    else // B.r < A.r
    {
        const int r = Br;
        C.U.SetType( GENERAL ); C.U.Resize( m, r );
        C.V.SetType( GENERAL ); C.V.Resize( n, r );

        // C.U C.V^T := alpha (A.U A.V^T)^H (B.U B.V^T)
        //            = alpha conj(A.V) A.U^H B.U B.V^T
        //            = (alpha conj(A.V) A.U^H B.U) B.V^T
        //            = conj(conj(alpha) A.V A.U^T conj(B.U)) B.V^T
        //            = conj(conj(alpha) A.V (B.U^H A.U)^T) B.V^T
        //            = conj(conj(alpha) A.V W^T) B.V^T
        //
        // W := B.U^H A.U
        // C.U := conj(alpha) A.V W^T
        // C.U := conj(C.U)
        // C.V := B.V
        Dense<Scalar> W( Br, Ar );
        blas::Gemm
        ( 'C', 'N', Br, Ar, B.Height(),
          1, B.U.LockedBuffer(), B.U.LDim(),
             A.U.LockedBuffer(), A.U.LDim(),
          0, W.Buffer(),         W.LDim() );
        blas::Gemm
        ( 'N', 'T', A.Width(), Br, Ar,
          Conj(alpha), A.V.LockedBuffer(), A.V.LDim(),
                       W.LockedBuffer(),   W.LDim(),
          0,           C.U.Buffer(),       C.U.LDim() );
        Conjugate( C.U );
        Copy( B.V, C.V );
    }
}

// Form a low-rank matrix from a dense matrix times a low-rank matrix
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const Dense<Scalar>& A,
                const LowRank<Scalar>& B,
                      LowRank<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::AdjointMultiply (F := D^H F)");
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Width();
    const int n = B.Width();
    const int r = B.Rank();

    C.U.SetType( GENERAL ); C.U.Resize( m, r );
    C.V.SetType( GENERAL ); C.V.Resize( n, r );

    if( A.Symmetric() )
    {
        // C.U C.V^[T,H] := alpha A^H B.U B.V^[T,H]
        //                = (alpha conj(A) B.U) B.V^[T,H]
        //                = conj(conj(alpha) A conj(B.U)) B.V^[T,H]
        //
        // BUConj := conj(B.U)
        // C.U := conj(alpha) A BUConj
        // C.U := conj(C.U)
        // C.V := B.V
        Dense<Scalar> BUConj;
        Conjugate( B.U, BUConj );
        blas::Symm
        ( 'L', 'L', B.Height(), r,
          Conj(alpha), A.LockedBuffer(),      A.LDim(),
                       BUConj.LockedBuffer(), BUConj.LDim(),
          0,           C.U.Buffer(),          C.U.LDim() );
        Conjugate( C.U );
        Copy( B.V, C.V );
    }
    else
    {
        // C.U C.V^[T,H] := alpha A^H B.U B.V^[T,H]
        //                = (alpha A^H B.U) B.V^[T,H]
        blas::Gemm
        ( 'C', 'N', m, r, A.Height(),
          alpha, A.LockedBuffer(),   A.LDim(),
                 B.U.LockedBuffer(), B.U.LDim(),
          0,     C.U.Buffer(),       C.U.LDim() );
        Copy( B.V, C.V );
    }
}

// Form a low-rank matrix from a low-rank matrix times a dense matrix
template<typename Scalar>
void AdjointMultiply
( Scalar alpha, const LowRank<Scalar>& A,
                const Dense<Scalar>& B,
                      LowRank<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::AdjointMultiply (F := F^H D)");
    if( A.Height() != B.Height() )
        throw std::logic_error("Cannot multiply nonconformal matrices.");
#endif
    const int m = A.Width();
    const int n = B.Width();
    const int r = A.Rank();

    C.U.SetType( GENERAL ); C.U.Resize( m, r );
    C.V.SetType( GENERAL ); C.V.Resize( n, r );

    // C.U C.V^T := alpha (A.U A.V^T)^H B
    //            = alpha conj(A.V) A.U^H B
    //            = conj(A.V) (alpha B^T conj(A.U))^T
    //
    // C.U := conj(A.V)
    // AUConj := conj(A.U)
    // C.V := alpha B^T AUConj
    Conjugate( A.V, C.U );
    Dense<Scalar> AUConj;
    Conjugate( A.U, AUConj );
    if( B.Symmetric() )
    {
        blas::Symm
        ( 'L', 'L', A.Height(), r,
          alpha, B.LockedBuffer(),      B.LDim(),
                 AUConj.LockedBuffer(), AUConj.LDim(),
          0,     C.V.Buffer(),          C.V.LDim() );
    }
    else
    {
        blas::Gemm
        ( 'T', 'N', n, r, A.Height(),
          alpha, B.LockedBuffer(),      B.LDim(),
                 AUConj.LockedBuffer(), AUConj.LDim(),
          0,     C.V.Buffer(),          C.V.LDim() );
    }
}

template<typename Real>
void AdjointMultiply
( int maxRank, std::complex<Real> alpha,
  const Dense<std::complex<Real> >& A,
  const Dense<std::complex<Real> >& B,
        LowRank<std::complex<Real> >& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::AdjointMultiply (F := D^H D)");
#endif
    typedef std::complex<Real> Scalar;

    const int m = A.Width();
    const int n = B.Width();
    const int minDim = std::min( m, n );
    const int r = std::min( minDim, maxRank );

    // C.U := alpha A^H B
    AdjointMultiply( alpha, A, B, C.U );

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

// Dense C := alpha A^H B
template void AdjointMultiply
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
                     Dense<float>& C );
template void AdjointMultiply
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
                      Dense<double>& C );
template void AdjointMultiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
                                   Dense<std::complex<float> >& C );
template void AdjointMultiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
                                    Dense<std::complex<double> >& C );

// Dense C := alpha A^H B + beta C
template void AdjointMultiply
( float alpha, const Dense<float>& A,
               const Dense<float>& B,
  float beta,        Dense<float>& C );
template void AdjointMultiply
( double alpha, const Dense<double>& A,
                const Dense<double>& B,
  double beta,        Dense<double>& C );
template void AdjointMultiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
  std::complex<float> beta,        Dense<std::complex<float> >& C );
template void AdjointMultiply
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              const Dense<std::complex<double> >& B,
  std::complex<double> beta,        Dense<std::complex<double> >& C );

// Form a dense matrix from a dense matrix times a low-rank matrix
template void AdjointMultiply
( float alpha, const Dense<float>& A,
               const LowRank<float>& B,
  float beta,        Dense<float>& C );
template void AdjointMultiply
( double alpha, const Dense<double>& A,
                const LowRank<double>& B,
  double beta,        Dense<double>& C );
template void AdjointMultiply
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             const LowRank<std::complex<float> >& B,
  std::complex<float> beta,        Dense<std::complex<float> >& C );
template void AdjointMultiply
( std::complex<double> alpha,
  const Dense<std::complex<double> >& A,
  const LowRank<std::complex<double> >& B,
  std::complex<double> beta,
        Dense<std::complex<double> >& C );

// Form a dense matrix from a low-rank matrix times a dense matrix
template void AdjointMultiply
( float alpha, const LowRank<float>& A,
               const Dense<float>& B,
  float beta,        Dense<float>& C );
template void AdjointMultiply
( double alpha, const LowRank<double>& A,
                const Dense<double>& B,
  double beta,        Dense<double>& C );
template void AdjointMultiply
( std::complex<float> alpha, const LowRank<std::complex<float> >& A,
                             const Dense<std::complex<float> >& B,
  std::complex<float> beta,        Dense<std::complex<float> >& C );
template void AdjointMultiply
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const Dense<std::complex<double> >& B,
  std::complex<double> beta,
        Dense<std::complex<double> >& C );

// Update a dense matrix from the product of two low-rank matrices
template void AdjointMultiply
( float alpha, const LowRank<float>& A,
               const LowRank<float>& B,
  float beta,        Dense<float>& C );
template void AdjointMultiply
( double alpha, const LowRank<double>& A,
                const LowRank<double>& B,
  double beta,        Dense<double>& C );
template void AdjointMultiply
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const LowRank<std::complex<float> >& B,
  std::complex<float> beta,
        Dense<std::complex<float> >& C );
template void AdjointMultiply
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const LowRank<std::complex<double> >& B,
  std::complex<double> beta,
        Dense<std::complex<double> >& C );

// Low-rank C := alpha A^H B
template void AdjointMultiply
( float alpha, const LowRank<float>& A,
               const LowRank<float>& B,
                     LowRank<float>& C );
template void AdjointMultiply
( double alpha, const LowRank<double>& A,
                const LowRank<double>& B,
                      LowRank<double>& C );
template void AdjointMultiply
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const LowRank<std::complex<float> >& B,
        LowRank<std::complex<float> >& C );
template void AdjointMultiply
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const LowRank<std::complex<double> >& B,
        LowRank<std::complex<double> >& C );

// Form a low-rank matrix from a dense matrix times a low-rank matrix
template void AdjointMultiply
( float alpha, const Dense<float>& A,
               const LowRank<float>& B,
                     LowRank<float>& C );
template void AdjointMultiply
( double alpha, const Dense<double>& A,
                const LowRank<double>& B,
                      LowRank<double>& C );
template void AdjointMultiply
( std::complex<float> alpha,
  const Dense<std::complex<float> >& A,
  const LowRank<std::complex<float> >& B,
        LowRank<std::complex<float> >& C );
template void AdjointMultiply
( std::complex<double> alpha,
  const Dense<std::complex<double> >& A,
  const LowRank<std::complex<double> >& B,
        LowRank<std::complex<double> >& C );

// Form a low-rank matrix from a low-rank matrix times a dense matrix
template void AdjointMultiply
( float alpha, const LowRank<float>& A,
               const Dense<float>& B,
                     LowRank<float>& C );
template void AdjointMultiply
( double alpha, const LowRank<double>& A,
                const Dense<double>& B,
                      LowRank<double>& C );
template void AdjointMultiply
( std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  const Dense<std::complex<float> >& B,
        LowRank<std::complex<float> >& C );
template void AdjointMultiply
( std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  const Dense<std::complex<double> >& B,
        LowRank<std::complex<double> >& C );

} // namespace hmat_tools
} // namespace dmhm

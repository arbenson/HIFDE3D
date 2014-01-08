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

// C :~= alpha A + beta B
//
// TODO:
// We could make use of a pivoted QLP factorization [Stewart, 1999] in order to
// get a more accurate approximation to the truncated singular value
// decomposition in O(mnk) work for an m x n matrix and k singular vectors.
//
// See Huckaby and Chan's 2004 paper:
// "Stewart's pivoted QLP decomposition for low-rank matrices".
//
// The current implementation attempts to pack as many of the needed buffers
// into one place and minimize data movement and flops as much as possible
// while still using BLAS3. This can be considered as most of the work towards
// almost entirely avoiding memory allocation since we could keep a sufficiently
// large buffer lying around and pack into it instead. This approach might be
// overly complicated, but rounded addition is supposedly one of the most
// expensive parts of H-algebra.

template<typename Real>
void RoundedAdd
( int maxRank,
  Real alpha, const LowRank<Real>& A,
  Real beta,  const LowRank<Real>& B,
                    LowRank<Real>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::AddRounded (F := F + F)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Incompatible matrix dimensions");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int minDim = std::min(m,n);
    const int Ar = A.Rank();
    const int Br = B.Rank();
    const int r = Ar + Br;
    const int roundedRank = std::min( std::min(r,minDim), maxRank );

    C.U.SetType( GENERAL ); C.U.Resize( m, roundedRank );
    C.V.SetType( GENERAL ); C.V.Resize( n, roundedRank );

    // Early exit if possible
    if( roundedRank == r )
    {
        // Copy alpha A.U into the left half of C.U
        for( int j=0; j<Ar; ++j )
        {
            Real* RESTRICT CUACol = C.U.Buffer(0,j);
            const Real* RESTRICT AUCol = A.U.LockedBuffer(0,j);
            for( int i=0; i<m; ++i )
                CUACol[i] = alpha*AUCol[i];
        }
        // Copy beta B.U into the right half of C.U
        for( int j=0; j<Br; ++j )
        {
            Real* RESTRICT CUBCol = C.U.Buffer(0,j+Ar);
            const Real* RESTRICT BUCol = B.U.LockedBuffer(0,j);
            for( int i=0; i<m; ++i )
                CUBCol[i] = beta*BUCol[i];
        }

        // Copy A.V into the left half of C.V
        for( int j=0; j<Ar; ++j )
            MemCopy( C.V.Buffer(0,j), A.V.LockedBuffer(0,j), n );

        // Copy B.V into the right half of C.V
        for( int j=0; j<Br; ++j )
            MemCopy( C.V.Buffer(0,j+Ar), B.V.LockedBuffer(0,j), n );

        return;
    }


    // Form U := [(alpha A.U), (beta B.U)]
    Dense<Real> U( m, r );
    for( int j=0; j<Ar; ++j )
    {
        Real* RESTRICT packedAUCol = U.Buffer(0,j);
        const Real* RESTRICT AUCol = A.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            packedAUCol[i] = alpha*AUCol[i];
    }
    for( int j=0; j<Br; ++j )
    {
        Real* RESTRICT packedBUCol = U.Buffer(0,j+Ar);
        const Real* RESTRICT BUCol = B.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            packedBUCol[i] = beta*BUCol[i];
    }

    // Form V := [A.V B.V]
    Dense<Real> V( n, r );
    for( int j=0; j<Ar; ++j )
        MemCopy( V.Buffer(0,j), A.V.LockedBuffer(0,j), n );
    for( int j=0; j<Br; ++j )
        MemCopy( V.Buffer(0,j+Ar), B.V.LockedBuffer(0,j), n );

#if defined(PIVOTED_QR)
    // TODO
    throw std::logic_error("Pivoted QR is not yet supported for this routine.");
#else
    // Perform an unpivoted QR decomposition on U
    const int minDimU = std::min(m,r);
    std::vector<Real> tauU( minDimU );
    std::vector<Real> workU( std::max(1,std::max(1,m)*r) );
    lapack::QR( m, r, U.Buffer(), U.LDim(), &tauU[0], &workU[0], workU.size() );

    // Perform an unpivoted QR decomposition on V
    const int minDimV = std::min(n,r);
    std::vector<Real> tauV( minDimV );
    std::vector<Real> workV( std::max(1,std::max(1,n)*r) );
    lapack::QR( n, r, V.Buffer(), V.LDim(), &tauV[0], &workV[0], workV.size() );

    // Copy R1 (the left factor's R from QR) into a zeroed buffer
    workU.resize( r*r );
    MemZero( &workU[0], r*r );
    for( int j=0; j<r; ++j )
        MemCopy( &workU[j*r], U.LockedBuffer(0,j), std::min(m,j+1) );

    // Copy R2 (the right factor's R from QR) into a zeroed buffer
    workV.resize( r*r );
    MemZero( &workV[0], r*r );
    for( int j=0; j<r; ++j )
        MemCopy( &workV[j*r], V.LockedBuffer(0,j), std::min(n,j+1) );

    // Form W := R1 R2^T.
    Dense<Real> W( minDimU, minDimV );
    blas::Gemm
    ( 'N', 'T', minDimU, minDimV, r,
      (Real)1, &workU[0], r, &workV[0], r, (Real)0, W.Buffer(), W.LDim() );

    // Get the SVD of R1 R2^T, overwriting R1 R2^T with UNew
    std::vector<Real> s( std::min(minDimU,minDimV) );
    Dense<Real> VT( std::min(minDimU,minDimV), minDimV );
    const int lworkSVD = lapack::SVDWorkSize( minDimU, minDimV );
    std::vector<Real> workSVD( lworkSVD );
    lapack::SVD
    ( 'O', 'S', minDimU, minDimV, W.Buffer(), W.LDim(),
      &s[0], 0, 1, VT.Buffer(), VT.LDim(), &workSVD[0], lworkSVD );

    // Form the rounded C.U by first filling it with
    //  | S*U_Left |, and then hitting it from the left with Q1
    //  |  0       |
    Scale( (Real)0, C.U );
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = s[j];
        Real* RESTRICT UColScaled = C.U.Buffer(0,j);
        const Real* RESTRICT UCol = W.LockedBuffer(0,j);
        for( int i=0; i<minDimU; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Apply Q1
    workU.resize( std::max(1,m*roundedRank) );
    lapack::ApplyQ
    ( 'L', 'N', m, roundedRank, minDimU, U.LockedBuffer(), U.LDim(), &tauU[0],
      C.U.Buffer(), C.U.LDim(), &workU[0], workU.size() );

    // Form the rounded C.V by first filling it with
    //  | (VT_Top)^T |, and then hitting it from the left with Q2
    //  |      0     |
    Scale( (Real)0, C.V );
    const int VTLDim = VT.LDim();
    for( int j=0; j<roundedRank; ++j )
    {
        Real* RESTRICT VCol = C.V.Buffer(0,j);
        const Real* RESTRICT VTRow = VT.LockedBuffer(j,0);
        for( int i=0; i<minDimV; ++i )
            VCol[i] = VTRow[i*VTLDim];
    }
    // Apply Q2
    workV.resize( std::max(1,n*roundedRank) );
    lapack::ApplyQ
    ( 'L', 'N', n, roundedRank, minDimV, V.LockedBuffer(), V.LDim(), &tauV[0],
      C.V.Buffer(), C.V.LDim(), &workV[0], workV.size() );
#endif // PIVOTED_QR
}

template<typename Real>
void RoundedAdd
( int maxRank,
  std::complex<Real> alpha,
  const LowRank<std::complex<Real> >& A,
  std::complex<Real> beta,
  const LowRank<std::complex<Real> >& B,
        LowRank<std::complex<Real> >& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::AddRounded (F := F + F)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Incompatible matrix dimensions");
#endif
    typedef std::complex<Real> Scalar;

    const int m = A.Height();
    const int n = A.Width();
    const int minDim = std::min(m,n);
    const int Ar = A.Rank();
    const int Br = B.Rank();
    const int r = Ar + Br;
    const int roundedRank = std::min( r, std::min(maxRank,minDim) );

    C.U.SetType( GENERAL ); C.U.Resize( m, roundedRank );
    C.V.SetType( GENERAL ); C.V.Resize( n, roundedRank );

    // Early exit if possible
    if( roundedRank == r )
    {
        // Copy alpha A.U into the left half of C.U
        for( int j=0; j<Ar; ++j )
        {
            Scalar* RESTRICT CUACol = C.U.Buffer(0,j);
            const Scalar* RESTRICT AUCol = A.U.LockedBuffer(0,j);
            for( int i=0; i<m; ++i )
                CUACol[i] = alpha*AUCol[i];
        }
        // Copy beta B.U into the right half of C.U
        for( int j=0; j<Br; ++j )
        {
            Scalar* RESTRICT CUBCol = C.U.Buffer(0,j+Ar);
            const Scalar* RESTRICT BUCol = B.U.LockedBuffer(0,j);
            for( int i=0; i<m; ++i )
                CUBCol[i] = beta*BUCol[i];
        }

        // Copy A.V into the left half of C.V
        for( int j=0; j<Ar; ++j )
            MemCopy( C.V.Buffer(0,j), A.V.LockedBuffer(0,j), n );

        // Copy B.V into the right half of C.V
        for( int j=0; j<Br; ++j )
            MemCopy( C.V.Buffer(0,j+Ar), B.V.LockedBuffer(0,j), n );

        return;
    }

    // Form U := [(alpha A.U) (beta B.U)]
    Dense<Scalar> U( m, r );
    for( int j=0; j<Ar; ++j )
    {
        Scalar* RESTRICT packedAUCol = U.Buffer(0,j);
        const Scalar* RESTRICT AUCol = A.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            packedAUCol[i] = alpha*AUCol[i];
    }
    for( int j=0; j<Br; ++j )
    {
        Scalar* RESTRICT packedBUCol = U.Buffer(0,j+Ar);
        const Scalar* RESTRICT BUCol = B.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            packedBUCol[i] = beta*BUCol[i];
    }

    // Form V := [A.V B.V]
    Dense<Scalar> V( n, r );
    for( int j=0; j<Ar; ++j )
        MemCopy( V.Buffer(0,j), A.V.LockedBuffer(0,j), n );

    for( int j=0; j<Br; ++j )
        MemCopy( V.Buffer(0,j+Ar), B.V.LockedBuffer(0,j), n );

#if defined(PIVOTED_QR)
    // TODO
    throw std::logic_error("Pivoted QR is not yet supported for this routine.");
#else
    // Perform an unpivoted QR decomposition on U
    const int minDimU = std::min(m,r);
    std::vector<Scalar> tauU( minDimU );
    std::vector<Scalar> workU( std::max(1,std::max(1,m)*r) );
    lapack::QR( m, r, U.Buffer(), U.LDim(), &tauU[0], &workU[0], workU.size() );

    // Perform an unpivoted QR decomposition on V
    const int minDimV = std::min(n,r);
    std::vector<Scalar> tauV( minDimV );
    std::vector<Scalar> workV( std::max(1,std::max(1,n)*r) );
    lapack::QR( n, r, V.Buffer(), V.LDim(), &tauV[0], &workV[0], workV.size() );

    // Copy R1 (the left factor's R from QR) into a zeroed buffer
    workU.resize( r*r );
    MemZero( &workU[0], r*r );
    for( int j=0; j<r; ++j )
        MemCopy( &workU[j*r], U.LockedBuffer(0,j), std::min(m,j+1) );

    // Copy R2 (the right factor's R from QR) into a zeroed buffer
    workV.resize( r*r );
    MemZero( &workV[0], r*r );
    for( int j=0; j<r; ++j )
        MemCopy( &workV[j*r], V.LockedBuffer(0,j), std::min(n,j+1) );

    // Form W := R1 R2^[T,H]
    const char option = 'T';
    Dense<Scalar> W( minDimU, minDimV );
    blas::Gemm
    ( 'N', option, minDimU, minDimV, r,
      Scalar(1), &workU[0], r, &workV[0], r, Scalar(0), W.Buffer(), W.LDim() );

    // Get the SVD of R1 R2^[T,H], overwriting R1 R2^[T,H] with UNew
    std::vector<Real> s( std::min(minDimU,minDimV) );
    Dense<Scalar> VH( std::min(minDimU,minDimV), minDimV );
    const int lworkSVD = lapack::SVDWorkSize( minDimU, minDimV );
    std::vector<Scalar> workSVD( lworkSVD );
    std::vector<Real> realWorkSVD( 5*r );
    lapack::SVD
    ( 'O', 'S', minDimU, minDimV, W.Buffer(), W.LDim(),
      &s[0], 0, 1, VH.Buffer(), VH.LDim(), &workSVD[0], lworkSVD,
      &realWorkSVD[0] );

    // Form the rounded C.U by first filling it with
    //  | S*U_Left |, and then hitting it from the left with Q1
    //  |  0       |
    Scale( Scalar(0), C.U );
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = s[j];
        Scalar* RESTRICT UColScaled = C.U.Buffer(0,j);
        const Scalar* RESTRICT UCol = W.Buffer(0,j);
        for( int i=0; i<minDimU; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Apply Q1
    workU.resize( std::max(1,m*roundedRank) );
    lapack::ApplyQ
    ( 'L', 'N', m, roundedRank, minDimU, U.LockedBuffer(), U.LDim(), &tauU[0],
      C.U.Buffer(), C.U.LDim(), &workU[0], workU.size() );

    // Form the rounded C.V by first filling it with
    //  | (VH_Top)^[T,H] |, and then hitting it from the left with Q2
    //  |      0         |
    Scale( Scalar(0), C.V );
    const int VHLDim = VH.LDim();
    for( int j=0; j<roundedRank; ++j )
    {
        Scalar* RESTRICT VColConj = C.V.Buffer(0,j);
        const Scalar* RESTRICT VHRow = VH.LockedBuffer(j,0);
        for( int i=0; i<minDimV; ++i )
            VColConj[i] = VHRow[i*VHLDim];
    }
    // Apply Q2
    workV.resize( std::max(1,n*roundedRank) );
    lapack::ApplyQ
    ( 'L', 'N', n, roundedRank, minDimV, V.LockedBuffer(), V.LDim(), &tauV[0],
      C.V.Buffer(), C.V.LDim(), &workV[0], workV.size() );
#endif // PIVOTED_QR
}

template void RoundedAdd
( int maxRank,
  float alpha, const LowRank<float>& A,
  float beta,  const LowRank<float>& B,
                     LowRank<float>& C );
template void RoundedAdd
( int maxRank,
  double alpha, const LowRank<double>& A,
  double beta,  const LowRank<double>& B,
                      LowRank<double>& C );
template void RoundedAdd
( int maxRank,
  std::complex<float> alpha,
  const LowRank<std::complex<float> >& A,
  std::complex<float> beta,
  const LowRank<std::complex<float> >& B,
        LowRank<std::complex<float> >& C );
template void RoundedAdd
( int maxRank,
  std::complex<double> alpha,
  const LowRank<std::complex<double> >& A,
  std::complex<double> beta,
  const LowRank<std::complex<double> >& B,
        LowRank<std::complex<double> >& C );

} // namespace hmat_tools
} // namespace hifde3d

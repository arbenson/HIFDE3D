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

// Compress a dense matrix into a low-rank matrix with specified rank
template<typename Real>
void Compress( int maxRank, Dense<Real>& D, LowRank<Real>& F )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Compress (Dense,LowRank)");
#endif
    const int m = D.Height();
    const int n = D.Width();
    const int minDim = std::min( m, n );
    const int r = std::min( minDim, maxRank );

    // Get the economic SVD of D, D = U Sigma V^T, overwriting F.U with U.
    F.U.Resize( m, minDim );
    Vector<Real> s( minDim );
    Dense<Real> VT( minDim, n );
    const int lwork = lapack::SVDWorkSize( m, n );
    std::vector<Real> work( lwork );
    lapack::SVD
    ( 'S', 'S', m, n, D.Buffer(), D.LDim(),
      s.Buffer(), F.U.Buffer(), F.U.LDim(), VT.Buffer(), VT.LDim(),
      &work[0], lwork );

    // Truncate the SVD in-place
    F.U.Resize( m, r );
    s.Resize( r );
    VT.Resize( r, n );

    // Put (Sigma V^T)^T = V Sigma into F.V
    F.V.Resize( n, r );
    const int VTLDim = VT.LDim();
    for( int j=0; j<r; ++j )
    {
        const Real sigma = s.Get(j);
        Real* RESTRICT VCol = F.V.Buffer(0,j);
        const Real* RESTRICT VTRow = VT.LockedBuffer(j,0);
        for( int i=0; i<n; ++i )
            VCol[i] = sigma*VTRow[i*VTLDim];
    }
}

template<typename Real>
void Compress
( int maxRank,
  Dense<std::complex<Real> >& D,
  LowRank<std::complex<Real> >& F )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Compress (Dense,LowRank)");
#endif
    typedef std::complex<Real> Scalar;

    const int m = D.Height();
    const int n = D.Width();
    const int minDim = std::min( m, n );
    const int r = std::min( minDim, maxRank );

    // Get the economic SVD of D, D = U Sigma V^H, overwriting F.U with U.
    F.U.Resize( m, minDim );
    Vector<Real> s( minDim );
    Dense<Scalar> VH( minDim, n );
    const int lwork = lapack::SVDWorkSize( m, n );
    std::vector<Scalar> work( lwork );
    std::vector<Real> rwork( 5*minDim );
    lapack::SVD
    ( 'S', 'S', m, n, D.Buffer(), D.LDim(),
      s.Buffer(), F.U.Buffer(), F.U.LDim(), VH.Buffer(), VH.LDim(),
      &work[0], lwork, &rwork[0] );

    // Truncate the SVD in-place
    F.U.Resize( m, r );
    s.Resize( r );
    VH.Resize( r, n );

    F.V.Resize( n, r );
    // Put (Sigma V^H)^T = conj(V) Sigma into F.V
    const int VHLDim = VH.LDim();
    for( int j=0; j<r; ++j )
    {
        const Real sigma = s.Get(j);
        Scalar* RESTRICT VCol = F.V.Buffer(0,j);
        const Scalar* RESTRICT VHRow = VH.LockedBuffer(j,0);
        for( int i=0; i<n; ++i )
            VCol[i] = sigma*VHRow[i*VHLDim];
    }
}

// A :~= A
//
// Approximate A with a given maximum rank.
template<typename Real>
void Compress( int maxRank, LowRank<Real>& A )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Compress (LowRank)");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int r = A.Rank();
    const int roundedRank = std::min( r, maxRank );
    if( roundedRank == r )
        return;

    // Grab enough workspace for our entire rounded addition
    const int leftPanelSize = std::max(1,std::max(1,m)*r);
    const int rightPanelSize = std::max(1,std::max(1,n)*r);
    const int blockSize = std::max(1,r*r);
    const int lworkSVD = lapack::SVDWorkSize( r, r );
    std::vector<Real> buffer
    ( 2*blockSize+std::max(lworkSVD,std::max(leftPanelSize,rightPanelSize)) );

#if defined(PIVOTED_QR)
    // TODO
    throw std::logic_error("Pivoted QR is not yet supported for this routine.");
#else
    // Perform an unpivoted QR decomposition on A.U
    std::vector<Real> tauU( std::min( m, r ) );
    lapack::QR
    ( m, r, A.U.Buffer(), A.U.LDim(), &tauU[0], &buffer[0], leftPanelSize );

    //------------------------------------------------------------------------//
    // buffer is logically empty                                              //
    //------------------------------------------------------------------------//

    // Perform an unpivoted QR decomposition on A.V
    std::vector<Real> tauV( std::min( n, r ) );
    lapack::QR
    ( n, r, A.V.Buffer(), A.V.LDim(), &tauV[0], &buffer[0], rightPanelSize );

    //------------------------------------------------------------------------//
    // buffer is logically empty                                              //
    //------------------------------------------------------------------------//

    // Copy R1 (the left factor's R from QR) into a zeroed buffer
    {
        MemZero( &buffer[0], blockSize );
        for( int j=0; j<r; ++j )
            MemCopy( &buffer[j*r], A.U.LockedBuffer(0,j), j );
    }

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize): R1                                                     //
    //------------------------------------------------------------------------//

    // Update W := R1 R2^T. We are unfortunately performing 2x as many
    // flops as are required.
    blas::Trmm
    ( 'R', 'U', 'T', 'N', r, r,
      1, A.V.LockedBuffer(), A.V.LDim(), &buffer[0], r );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize): R1 R2^T                                                //
    //------------------------------------------------------------------------//

    // Get the SVD of R1 R2^T, overwriting R1 R2^T with U
    std::vector<Real> s( r );
    lapack::SVD
    ( 'O', 'S', r, r, &buffer[0], r, &s[0], 0, 1,
      &buffer[blockSize], r, &buffer[2*blockSize], lworkSVD );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize):           U of R1 R2^T                                 //
    //  [blockSize,2*blockSize): V^T of R1 R2^T                               //
    //------------------------------------------------------------------------//

    // Copy the result of the QR factorization of A.U into a temporary buffer
    for( int j=0; j<r; ++j )
        MemCopy( &buffer[2*blockSize+j*m], A.U.LockedBuffer(0,j), m );
    // Logically shrink A.U
    A.U.Resize( m, roundedRank );
    // Zero the shrunk buffer
    Scale( (Real)0, A.U );
    // Copy the scaled U from the SVD of R1 R2^T into the top of the matrix
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = s[j];
        const Real* RESTRICT UCol = &buffer[j*r];
        Real* RESTRICT UColScaled = A.U.Buffer(0,j);
        for( int i=0; i<r; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Hit the matrix from the left with Q1 from the QR decomp of the orig A.U
    lapack::ApplyQ
    ( 'L', 'N', m, roundedRank, r, &buffer[2*blockSize], m, &tauU[0],
      A.U.Buffer(), A.U.LDim(), &buffer[0], blockSize );

    // Copy the result of the QR factorization of A.V into a temporary buffer
    for( int j=0; j<r; ++j )
        MemCopy( &buffer[2*blockSize+j*n], A.V.LockedBuffer(0,j), n );
    // Logically shrink A.V
    A.V.Resize( n, roundedRank );
    // Zero the shrunk buffer
    Scale( (Real)0, A.V );
    // Copy V=(V^T)^T from the SVD of R1 R2^T into the top of A.V
    for( int j=0; j<roundedRank; ++j )
    {
        const Real* RESTRICT VTRow = &buffer[blockSize+j];
        Real* RESTRICT VCol = A.V.Buffer(0,j);
        for( int i=0; i<r; ++i )
            VCol[i] = VTRow[i*r];
    }
    // Hit the matrix from the left with Q2 from the QR decomp of the orig A.V
    lapack::ApplyQ
    ( 'L', 'N', n, roundedRank, r, &buffer[2*blockSize], n, &tauV[0],
      A.V.Buffer(), A.V.LDim(), &buffer[0], blockSize );
#endif // PIVOTED_QR
}

// A :~= A
//
// Approximate A with a given maximum rank.
template<typename Real>
void Compress( int maxRank, LowRank<std::complex<Real> >& A )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Compress (LowRank)");
#endif
    typedef std::complex<Real> Scalar;

    const int m = A.Height();
    const int n = A.Width();
    const int r = A.Rank();
    const int roundedRank = std::min( r, maxRank );
    if( roundedRank == r )
        return;

    // Grab enough workspace for our entire rounded addition
    const int leftPanelSize = std::max(1,std::max(1,m)*r);
    const int rightPanelSize = std::max(1,std::max(1,n)*r);
    const int blockSize = std::max(1,r*r);
    const int lworkSVD = lapack::SVDWorkSize( r, r );
    std::vector<Scalar> buffer
    ( 2*blockSize+std::max(lworkSVD,std::max(leftPanelSize,rightPanelSize)) );

#if defined(PIVOTED_QR)
    // TODO
    throw std::logic_error("Pivoted QR is not yet supported for this routine.");
#else
    // Perform an unpivoted QR decomposition on A.U
    std::vector<Scalar> tauU( std::min( m, r ) );
    lapack::QR
    ( m, r, A.U.Buffer(), A.U.LDim(), &tauU[0], &buffer[0], leftPanelSize );

    //------------------------------------------------------------------------//
    // buffer is logically empty                                              //
    //------------------------------------------------------------------------//

    // Perform an unpivoted QR decomposition on A.V
    std::vector<Scalar> tauV( std::min( n, r ) );
    lapack::QR
    ( n, r, A.V.Buffer(), A.V.LDim(), &tauV[0], &buffer[0], rightPanelSize );

    //------------------------------------------------------------------------//
    // buffer is logically empty                                              //
    //------------------------------------------------------------------------//

    // Copy R1 (the left factor's R from QR) into a zeroed buffer
    {
        MemZero( &buffer[0], blockSize );
        for( int j=0; j<r; ++j )
            MemCopy( &buffer[j*r], A.U.LockedBuffer(0,j), j );
    }

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize): R1                                                     //
    //------------------------------------------------------------------------//

    // Update W := R1 R2^[T,H].
    // We are unfortunately performing 2x as many flops as required.
    const char option = 'T';
    blas::Trmm
    ( 'R', 'U', option, 'N', r, r,
      1, A.V.LockedBuffer(), A.V.LDim(), &buffer[0], r );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize): R1 R2^[T,H]                                            //
    //------------------------------------------------------------------------//

    // Get the SVD of R1 R2^[T,H], overwriting it with U
    std::vector<Real> realBuffer( 6*r );
    lapack::SVD
    ( 'O', 'S', r, r, &buffer[0], r, &realBuffer[0], 0, 1,
      &buffer[blockSize], r, &buffer[2*blockSize], lworkSVD, &realBuffer[r] );

    //------------------------------------------------------------------------//
    // buffer contains:                                                       //
    //  [0,blockSize):           U of R1 R2^[T,H]                             //
    //  [blockSize,2*blockSize): V^H of R1 R2^[T,H]                           //
    //                                                                        //
    // realBuffer contains:                                                   //
    //   [0,r): singular values of R1 R2^[T,H]                                //
    //------------------------------------------------------------------------//

    // Copy the result of the QR factorization of A.U into a temporary buffer
    for( int j=0; j<r; ++j )
        MemCopy( &buffer[2*blockSize+j*m], A.U.LockedBuffer(0,j), m );
    // Logically shrink A.U
    A.U.Resize( m, roundedRank );
    // Zero the shrunk buffer
    Scale( Scalar(0), A.U );
    // Copy the scaled U from the SVD of R1 R2^[T,H] into the top of the matrix
    for( int j=0; j<roundedRank; ++j )
    {
        const Real sigma = realBuffer[j];
        Scalar* RESTRICT UColScaled = A.U.Buffer(0,j);
        const Scalar* RESTRICT UCol = &buffer[j*r];
        for( int i=0; i<r; ++i )
            UColScaled[i] = sigma*UCol[i];
    }
    // Hit the matrix from the left with Q1 from the QR decomp of the orig A.U
    lapack::ApplyQ
    ( 'L', 'N', m, roundedRank, r, &buffer[2*blockSize], m, &tauU[0],
      A.U.Buffer(), A.U.LDim(), &buffer[0], blockSize );

    // Copy the result of the QR factorization of A.V into a temporary buffer
    for( int j=0; j<r; ++j )
        MemCopy( &buffer[2*blockSize+j*n], A.V.LockedBuffer(0,j), n );
    // Logically shrink A.V
    A.V.Resize( n, roundedRank );
    // Zero the shrunk buffer
    Scale( Scalar(0), A.V );
    // Copy conj(V)=(V^H)^T from the SVD of R1 R2^T into the top of A.V
    for( int j=0; j<roundedRank; ++j )
    {
        Scalar* RESTRICT VColConj = A.V.Buffer(0,j);
        const Scalar* RESTRICT VHRow = &buffer[blockSize+j];
        for( int i=0; i<r; ++i )
            VColConj[i] = VHRow[i*r];
    }
    // Hit the matrix from the left with Q2 from the QR decomp of the orig A.V
    lapack::ApplyQ
    ( 'L', 'N', n, roundedRank, r, &buffer[2*blockSize], n, &tauV[0],
      A.V.Buffer(), A.V.LDim(), &buffer[0], blockSize );
#endif // PIVOTED_QR
}

template void Compress( int maxRank, Dense<float>& D, LowRank<float>& F );
template void Compress( int maxRank, Dense<double>& D, LowRank<double>& F );
template void Compress
( int maxRank,
  Dense<std::complex<float> >& D,
  LowRank<std::complex<float> >& F );
template void Compress
( int maxRank,
  Dense<std::complex<double> >& D,
  LowRank<std::complex<double> >& F );

template void Compress( int maxRank, LowRank<float>& A );
template void Compress( int maxRank, LowRank<double>& A );
template void Compress( int maxRank, LowRank<std::complex<float> >& A );
template void Compress( int maxRank, LowRank<std::complex<double> >& A );

} // namespace hmat_tools
} // namespace dhmhm

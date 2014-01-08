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

// Dense B := alpha A + beta B
template<typename Scalar>
void Update
( Scalar alpha, const Dense<Scalar>& A,
  Scalar beta,        Dense<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Update (D := D + D)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Tried to update with nonconforming matrices.");
    // TODO: Allow for A to be symmetric when B is general
    if( A.Symmetric() && B.General() )
        throw std::logic_error("A-symmetric/B-general not yet implemented.");
    if( A.General() && B.Symmetric() )
        throw std::logic_error
        ("Cannot update a symmetric matrix with a general one");
#endif
    const int m = A.Height();
    const int n = A.Width();
    if( A.Symmetric() )
    {
        for( int j=0; j<n; ++j )
        {
            Scalar* RESTRICT BCol = B.Buffer(0,j);
            const Scalar* RESTRICT ACol = A.LockedBuffer(0,j);
            for( int i=j; i<m; ++i )
                BCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
    else
    {
        for( int j=0; j<n; ++j )
        {
            Scalar* RESTRICT BCol = B.Buffer(0,j);
            const Scalar* RESTRICT ACol = A.LockedBuffer(0,j);
            for( int i=0; i<m; ++i )
                BCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
}

// Low-rank B := alpha A + beta B
template<typename Scalar>
void Update
( Scalar alpha, const LowRank<Scalar>& A,
  Scalar beta,        LowRank<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Update (F := F + F)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Tried to update with nonconforming matrices.");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int Ar = A.Rank();
    const int Br = B.Rank();
    const int newRank = Ar + Br;

    // B.U := [(beta B.U), (alpha A.U)]
    Scale( beta, B.U );
    B.U.Resize( B.Height(), newRank );
    // Copy in (alpha A.U)
    for( int j=0; j<Ar; ++j )
    {
        Scalar* RESTRICT BUACol = B.U.Buffer(0,j+Br);
        const Scalar* RESTRICT AUCol = A.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            BUACol[i] = alpha*AUCol[i];
    }

    // B.V := [B.V A.V]
    B.V.Resize( B.Width(), newRank );
    for( int j=0; j<Ar; ++j )
        MemCopy( B.V.Buffer(0,j+Br), A.V.LockedBuffer(0,j), n );
}

// Dense updated with low-rank, B := alpha A + beta B
template<typename Scalar>
void Update
( Scalar alpha, const LowRank<Scalar>& A,
  Scalar beta,        Dense<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Update (D := F + D)");
    if( A.Height() != B.Height() || A.Width() != B.Width()  )
        throw std::logic_error("Tried to update with nonconforming matrices.");
    if( B.Symmetric() )
        throw std::logic_error("Unsafe update of symmetric dense matrix.");
#endif
    const char option = 'T';
    blas::Gemm
    ( 'N', option, A.Height(), A.Width(), A.Rank(),
      alpha, A.U.LockedBuffer(), A.U.LDim(), A.V.LockedBuffer(), A.V.LDim(),
      beta, B.Buffer(), B.LDim() );
}

// Dense update B := alpha A + beta B
template void Update
( float alpha, const Dense<float>& A,
  float beta,        Dense<float>& B );
template void Update
( double alpha, const Dense<double>& A,
  double beta,        Dense<double>& B );
template void Update
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
  std::complex<float> beta,        Dense<std::complex<float> >& B );
template void Update
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
  std::complex<double> beta,        Dense<std::complex<double> >& B );

// Low-rank update B := alpha A + beta B
template void Update
( float alpha, const LowRank<float>& A,
  float beta,        LowRank<float>& B );
template void Update
( double alpha, const LowRank<double>& A,
  double beta,        LowRank<double>& B );
template void Update
( std::complex<float> alpha, const LowRank<std::complex<float> >& A,
  std::complex<float> beta, LowRank<std::complex<float> >& B );
template void Update
( std::complex<double> alpha, const LowRank<std::complex<double> >& A,
  std::complex<double> beta, LowRank<std::complex<double> >& B );

// Dense updated with low-rank, B := alpha A + beta B
template void Update
( float alpha, const LowRank<float>& A,
  float beta,        Dense<float>& B );
template void Update
( double alpha, const LowRank<double>& A,
  double beta,        Dense<double>& B );
template void Update
( std::complex<float> alpha, const LowRank<std::complex<float> >& A,
  std::complex<float> beta, Dense<std::complex<float> >& B );
template void Update
( std::complex<double> alpha, const LowRank<std::complex<double> >& A,
  std::complex<double> beta, Dense<std::complex<double> >& B );

} // namespace hmat_tools
} // namespace hifde3d

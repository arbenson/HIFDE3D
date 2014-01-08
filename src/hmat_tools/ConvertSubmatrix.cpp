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

/*
template<typename Scalar>
void ConvertSubmatrix
( Dense<Scalar>& D, const Sparse<Scalar>& S,
  int iStart, int jStart, int height, int width )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::ConvertSubmatrix (Dense,Sparse)");
#endif
    // Initialize the dense matrix to all zeros
    if( S.symmetric && iStart == jStart )
        D.SetType( SYMMETRIC );
    else
        D.SetType( GENERAL );
    D.Resize( height, width );
    Scale( Scalar(0), D );
#ifndef RELEASE
    if( D.Symmetric() && height != width )
        throw std::logic_error("Invalid submatrix of symmetric sparse matrix.");
#endif

    // Add in the nonzeros, one row at a time
    const int ldim = D.LDim();
    Scalar* DBuffer = D.Buffer();
    for( int iOffset=0; iOffset<height; ++iOffset )
    {
        const int thisRowOffset = S.rowOffsets[iStart+iOffset];
        const int nextRowOffset = S.rowOffsets[iStart+iOffset+1];

        const int* thisSetOfColIndices = &S.columnIndices[thisRowOffset];
        for( int k=0; k<nextRowOffset-thisRowOffset; ++k )
        {
            const int thisColIndex = thisSetOfColIndices[k];
            if( thisColIndex < jStart || thisColIndex >= jStart+width )
                continue;
            else
            {
                const int jOffset = thisColIndex - jStart;
                DBuffer[iOffset+jOffset*ldim] = S.nonzeros[thisRowOffset+k];
            }
        }
    }
}

template<typename Scalar>
void ConvertSubmatrix
( LowRank<Scalar>& F, const Sparse<Scalar>& S,
  int iStart, int jStart, int height, int width )
{
#ifndef RELEASE
    CallStackEntry entry
    ("hmat_tools::ConvertSubmatrix (LowRank,Sparse)");
#endif
    // Figure out the matrix sizes
    int rankCounter = 0;
    for( int iOffset=0; iOffset<height; ++iOffset )
    {
        const int thisRowOffset = S.rowOffsets[iStart+iOffset];
        const int nextRowOffset = S.rowOffsets[iStart+iOffset+1];

        const int* thisSetOfColIndices = &S.columnIndices[thisRowOffset];
        for( int k=0; k<nextRowOffset-thisRowOffset; ++k )
        {
            const int thisColIndex = thisSetOfColIndices[k];
            if( thisColIndex < jStart || thisColIndex >= jStart+width)
                continue;
            else
                ++rankCounter;
        }
    }

    const int r = rankCounter;
    F.U.SetType( GENERAL ); F.U.Resize( height, r );
    F.V.SetType( GENERAL ); F.V.Resize( width, r );
    Scale( Scalar(0), F.U );
    Scale( Scalar(0), F.V );

    // Fill in the representation of each nonzero using the appropriate column
    // of identity in F.U and the appropriate scaled column of identity in
    // F.V
    rankCounter = 0;
    for( int iOffset=0; iOffset<height; ++iOffset )
    {
        const int thisRowOffset = S.rowOffsets[iStart+iOffset];
        const int nextRowOffset = S.rowOffsets[iStart+iOffset+1];

        const int* thisSetOfColIndices = &S.columnIndices[thisRowOffset];
        for( int k=0; k<nextRowOffset-thisRowOffset; ++k )
        {
            const int thisColIndex = thisSetOfColIndices[k];
            if( thisColIndex < jStart || thisColIndex >= jStart+width )
                continue;
            else
            {
                const int jOffset = thisColIndex - jStart;
                const Scalar value = S.nonzeros[thisRowOffset+k];
                F.U.Set(iOffset,rankCounter,1);
                F.V.Set(jOffset,rankCounter,value);
                ++rankCounter;
            }
        }
    }
#ifndef RELEASE
    if( F.Rank() > std::min(height,width) )
        std::logic_error("Rank is larger than minimum dimension");
#endif
}

template void ConvertSubmatrix
(       Dense<float>& D,
  const Sparse<float>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void ConvertSubmatrix
(       Dense<double>& D,
  const Sparse<double>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void ConvertSubmatrix
(       Dense<std::complex<float> >& D,
  const Sparse<std::complex<float> >& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void ConvertSubmatrix
(       Dense<std::complex<double> >& D,
  const Sparse<std::complex<double> >& S,
  int iStart, int iEnd, int jStart, int jEnd );

template void ConvertSubmatrix
(       LowRank<float>& F,
  const Sparse<float>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void ConvertSubmatrix
(       LowRank<double>& F,
  const Sparse<double>& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void ConvertSubmatrix
(       LowRank<std::complex<float> >& F,
  const Sparse<std::complex<float> >& S,
  int iStart, int iEnd, int jStart, int jEnd );
template void ConvertSubmatrix
(       LowRank<std::complex<double> >& F,
  const Sparse<std::complex<double> >& S,
  int iStart, int iEnd, int jStart, int jEnd );

*/
} // namespace hmat_tools
} // namespace hifde3d

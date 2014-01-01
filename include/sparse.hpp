/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_SPARSE_MATRIX_HPP
#define DMHM_SPARSE_MATRIX_HPP 1

#include "vector.hpp"

namespace dmhm {

// A simple Compressed Sparse Row (CSR) data structure
template<typename Scalar>
struct Sparse
{
    bool symmetric;
    int height, width;
    Vector<Scalar> nonzeros;
    Vector<int> columnIndices;
    Vector<int> rowOffsets;

    void Clear();

    void Print( const std::string tag, std::ostream& os=std::cout ) const;
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar>
inline void
Sparse<Scalar>::Clear()
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Clear");
#endif
    symmetric = false;
    height = 0;
    width = 0;
    nonzeros.Clear();
    columnIndices.Clear();
    rowOffsets.Clear();
}

template<typename Scalar>
inline void
Sparse<Scalar>::Print( const std::string tag, std::ostream& os ) const
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Print");
#endif
    if( symmetric )
        os << tag << "(symmetric)\n";
    else
        os << tag << "\n";

    for( int i=0; i<height; ++i )
    {
        const int numCols = rowOffsets[i+1]-rowOffsets[i];
        const int rowOffset = rowOffsets[i];
        for( int k=0; k<numCols; ++k )
        {
            const int j = columnIndices[rowOffset+k];
            const Scalar alpha = nonzeros[rowOffset+k];
            os << i << " " << j << " " << WrapScalar(alpha) << "\n";
        }
    }
    os << std::endl;
}

} // namespace dmhm

#endif // ifndef DMHM_SPARSE_MATRIX_HPP

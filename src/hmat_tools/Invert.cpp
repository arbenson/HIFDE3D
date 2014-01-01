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

template<typename Scalar>
void Invert( Dense<Scalar>& D )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Invert");
    if( D.Height() != D.Width() )
        throw std::logic_error("Tried to invert a non-square dense matrix.");
#endif
    const int n = D.Height();
    std::vector<int> ipiv( n );
    if( D.Symmetric() )
    {
        const int lworkLDLT = lapack::LDLTWorkSize( n );
        const int lworkInvertLDLT = lapack::InvertLDLTWorkSize( n );
        const int lwork = std::max( lworkLDLT, lworkInvertLDLT );
        std::vector<Scalar> work( lwork );

        lapack::LDLT( 'L', n, D.Buffer(), D.LDim(), &ipiv[0], &work[0], lwork );
        lapack::InvertLDLT( 'L', n, D.Buffer(), D.LDim(), &ipiv[0], &work[0] );
    }
    else
    {
        const int lwork = lapack::InvertLUWorkSize( n );
        std::vector<Scalar> work( lwork );

        lapack::LU( n, n, D.Buffer(), D.LDim(), &ipiv[0] );
        lapack::InvertLU( n, D.Buffer(), D.LDim(), &ipiv[0], &work[0], lwork );
    }
}

template void Invert( Dense<float>& D );
template void Invert( Dense<double>& D );
template void Invert( Dense<std::complex<float> >& D );
template void Invert( Dense<std::complex<double> >& D );

} // namespace hmat_tools
} // namespace dmhm

/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_LOW_RANK_HPP
#define DMHM_LOW_RANK_HPP 1

#include "dmhm/core/dense.hpp"

namespace dmhm {

// A basic low-rank matrix representation that is used for the blocks with
// sufficiently separated sources and targets.
//
// U and V will always be assumed to be of general type
// (they should be non-square except in pathological cases).
template<typename Scalar>
struct LowRank
{
    // A = U V^T.
    Dense<Scalar> U, V;

    int Height() const { return U.Height(); }
    int Width() const { return V.Height(); }
    int Rank() const { return U.Width(); }

    void Print( const std::string tag, std::ostream& os=std::cout ) const
    {
        os << tag << ":\n";
        U.Print( "U", os );
        V.Print( "V", os );
    }
};

} // namespace dmhm

#endif // ifndef DMHM_LOW_RANK_HPP

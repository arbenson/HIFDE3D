#ifndef _ID_HPP_
#define _ID_HPP_

#include "dense.hpp"

#include <vector>

// Interpolative decomposition (ID) of a matrix M:
//      
//       M =~ M(:, skel) * W,
//
// where M(:, skel) is a subset of the columns of M and W is the
// interpolating factor.  The ID is computed using a column-pivoted
// QR factorization.
// 
// M (in): matrix on which to perform the ID.  M gets overwritten.
// W (out): matrix where the interpolating factor gets stored
// skeleton_cols (out): list of skeleton cols of the ID
// redundant_cols (out): list of redundant (non-skeleton) columns of the ID
// epsilon (in): tolerance for the ID
template <typename Scalar>
int InterpDecomp(Dense<Scalar>& M, const Dense<Scalar>& W,
		 std::vector<int>& skeleton_cols,
		 std::vector<int>& redundant_cols, double epsilon);

#endif  // _ID_HPP_

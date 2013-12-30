#ifndef _SCHUR_INTERIOR_HPP_
#define _SCHUR_INTERIOR_HPP_

#include <vector>

// Schur out DOFs from a matrix.  The matrix contains the DOFs to be eliminated
// and the interaction of these DOFs in the rest of the matrix.
// If the DOF set is size m and there are n interactions, then the matrix is of
// size (m + n) x (m + n).
// 
// matrix (in): dense matrix containing the DOF set to be eliminated and the
//              interaction of this DOF set.
// DOF_set (in): indices of the degrees of freedom
// DOF_set_interaction (in): indices of the interactions of DOF_set
// data (out): data to be filled A_22, A_22^{-1}, A22^{-1} * A21,
//             and Schur complement
// return value: 0 on failure, 1 on success
template<T>
int SchurInterior(NumMat<T> matrix, const std::vector<int>& DOF_set,
		  const std::vector<int>& DOF_set_interaction,
		  SchurData& data);


// For a given cell location at a given level, determine the indices of the
// DOFs interior to the cell.  These DOFs are eliminated by a Schur
// complement (see SchurInterior).
//
// cell_location (in): 3-tuple of cell location
// W (in): width of the cell
// N (in): number of discretization points in each direction
// remaining_DOFs (in): 3-dimensional array of remaining degrees of freedom
// DOF_set (out): indices of remaining degrees of freedom in the cell location
// DOF_set_interaction (out): interaction of the DOF set with other remaining
//                            degrees of freedom
// return value: 0 on failure, 1 on success
int GetDOFsAndInteractions(Index3 cell_location, int W, int N,
			   const IntNumTns& remaining_DOFs,
			   std::vector<int>& DOF_set,
			   std::vector<int>& DOF_set_interaction);

#endif  // _SCHUR_INTERIOR_HPP_

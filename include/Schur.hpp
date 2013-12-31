#ifndef SCHUR_HPP_
#define SCHUR_HPP_

#include "dense.hpp"
#include "sparse.hpp"
#include "vector.hpp"

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
template <typename Scalar>
int Schur(Dense<Scalar>& matrix, const Vector<int>& DOF_set,
	  const Vector<int>& DOF_set_interaction,
	  SchurData& data);

// Extract a dense submatrix from a sparse matrix.
//
// sp_matrix (in): sparse matrix from which to extract entries
// rows (in): row indices
// cols (in): column indices
// submatrix (out): sp_matrix(rows, cols) as a dense matrix
// return value: 0 on failure, 1 on success
template <typename Scalar>
int DenseSubmatrix(Sparse<T> sp_matrix, const Vector<int>& rows,
		   const Vector<int>& cols, const Dense<Scalar>& submatrix);

#endif  // ifndef SCHUR_HPP_

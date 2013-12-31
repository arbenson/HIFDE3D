#ifndef SCHUR_HPP_
#define SCHUR_HPP_

#include "data.hpp"
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
void Schur(Dense<Scalar>& matrix, FactorData<Scalar>& data) {
    auto DOF_set = data.ind_data().DOF_set();
    auto DOF_set_interaction = data.ind_data().DOF_set_interaction();

    Dense<Scalar> A12;
    DenseSubmatrix(matrix, DOF_set_interaction, DOF_set, A12);
    Dense<Scalar> A21;
    DenseSubmatrix(matrix, DOF_set, DOF_set_interaction, A21);
    DenseSubmatrix(matrix, DOF_set, DOF_set, data.A22());

    // TODO: implement the inverse function
    data.A22().Inverse(data.A22_inv());
    data.A22_inv().Multiply(One<Scalar>, A21, data.X_mat());
    A12.Multiply(NegativeOne<Scalar>, data.X_mat(), data.Schur_comp());
}

// Extract a dense submatrix from a sparse matrix.
// TODO: this function could be more efficient
//
// matrix (in): sparse matrix from which to extract entries
// rows (in): row indices
// cols (in): column indices
// submatrix (out): sp_matrix(rows, cols) as a dense matrix
// return value: 0 on failure, 1 on success
template <typename Scalar>
int DenseSubmatrix(Sparse<Scalar>& matrix, const Vector<int>& rows,
		   const Vector<int>& cols, const Dense<Scalar>& submatrix) {
    submatrix.Resize(rows.Size(), cols.Size());
    for (int i = 0; i < rows.Size(); ++i) {
	for (int j = 0; i < cols.Size(); ++i) {
	    submatrix.Set(i, j, matrix.Get(rows[i], cols[j]);
	}
    }
}

// Extract a dense submatrix from a dense matrix.
// TODO: this function could be more efficient
//
// matrix (in): dense matrix from which to extract entries
// rows (in): row indices
// cols (in): column indices
// submatrix (out): sp_matrix(rows, cols) as a dense matrix
// return value: 0 on failure, 1 on success
template <typename Scalar>
int DenseSubmatrix(Dense<Scalar>& matrix, const Vector<int>& rows,
		   const Vector<int>& cols, const Dense<Scalar>& submatrix) {
    submatrix.Resize(rows.Size(), cols.Size());
    for (int i = 0; i < rows.Size(); ++i) {
	for (int j = 0; i < cols.Size(); ++i) {
	    submatrix.Set(i, j, matrix.Get(rows[i], cols[j]);
	}
    }
}

#endif  // ifndef SCHUR_HPP_

#ifndef SCHUR_HPP_
#define SCHUR_HPP_

namespace hifde3d {

// Extract a dense submatrix from a dense matrix.
// TODO: this function could be more efficient
//
// matrix (in): dense matrix from which to extract entries
// rows (in): row indices
// cols (in): column indices
// submatrix (out): sp_matrix(rows, cols) as a dense matrix
template <typename Scalar>
void DenseSubmatrix(const Dense<Scalar>& matrix, const std::vector<int>& rows,
                    const std::vector<int>& cols, Dense<Scalar>& submatrix) {
#ifndef RELEASE
    CallStackEntry entry("DenseSubmatrix");
#endif
    submatrix.Resize(rows.size(), cols.size());
    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t j = 0; j < cols.size(); ++j) {
            submatrix.Set(i, j, matrix.Get(rows[i], cols[j]));
        }
    }
}

// Extract a dense submatrix from a sparse matrix.
//
// matrix (in): sparse matrix from which to extract entries
// rows (in): row indices
// cols (in): column indices
// submatrix (out): sp_matrix(rows, cols) as a dense matrix
template <typename Scalar>
void DenseSubmatrix(Sparse<Scalar>& matrix, std::vector<int>& rows,
                    std::vector<int>& cols, Dense<Scalar>& submatrix) {
#ifndef RELEASE
    CallStackEntry entry("DenseSubmatrix");
#endif
    // TODO: avoid this copy
    Vector<int> iidx(rows.size());
    for (size_t i = 0; i < rows.size(); ++i) {
	iidx.Set(i, rows[i]);
    }
    Vector<int> jidx(cols.size());
    for (size_t j = 0; j < cols.size(); ++j) {
	jidx.Set(j, cols[j]);
    }
    submatrix.Resize(iidx.Size(), jidx.Size());
    matrix.Find(iidx, jidx, submatrix);
}

// Schur out DOFs from a matrix.  The matrix contains the DOFs to be eliminated
// and the interaction of these DOFs in the rest of the matrix.
// If the DOF set is size m and there are n interactions, then the matrix is of
// size (m + n) x (m + n).
//
// matrix (in): dense matrix containing the DOF set to be eliminated and the
//              interaction of this DOF set.
// data (out): data to be filled A_22, A_22^{-1}, A_22^{-1} * A21,
//             and Schur complement
template <typename Scalar>
void Schur(Dense<Scalar>& matrix, FactorData<Scalar>& data) {
#ifndef RELEASE
    CallStackEntry entry("Schur");
#endif
    std::vector<int>& red_inds = data.ind_data().redundant_inds();
    std::vector<int>& skel_inds = data.ind_data().skeleton_inds();

    Dense<Scalar> A12;
    DenseSubmatrix(matrix, skel_inds, red_inds, A12);
    Dense<Scalar> A21;
    DenseSubmatrix(matrix, red_inds, skel_inds, A21);
    DenseSubmatrix(matrix, red_inds, red_inds, data.A_22());
    DenseSubmatrix(matrix, red_inds, red_inds, data.A_22_inv());
    // TODO: probably faster to copy A_22 into A_22_inv, rather than reading
    // from the matrix again
    hmat_tools::Invert(data.A_22_inv());

    // X = A_22^{-1}A_{21}
    // S = -A_{12}A_22^{-1}A_{21} = -A_{12}X
    // TODO: This was the notation in the Matlab code, but it makes more
    // sense to swap the 1 and 2.  That way, the Schur complement uses
    // A_{11}^{-1}.

    data.X_mat().Resize(data.A_22_inv().Height(), A21.Width());
    hmat_tools::Multiply(Scalar(1), data.A_22_inv(), A21, data.X_mat());
    data.Schur_comp().Resize(A12.Height(), data.X_mat().Width());
    hmat_tools::Multiply(Scalar(-1), A12, data.X_mat(), data.Schur_comp());
}

}
#endif  // ifndef SCHUR_HPP_

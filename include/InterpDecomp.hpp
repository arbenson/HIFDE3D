#ifndef INTERP_DECOMP_HPP_
#define INTERP_DECOMP_HPP_

namespace hifde3d {
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
void InterpDecomp(hifde3d::Dense<Scalar>& M, hifde3d::Dense<Scalar>& W,
                  std::vector<int>& skeleton_cols,
                  std::vector<int>& redundant_cols, double epsilon);

// Solve RX = B, where R is upper triangular.  Overwrite B with X.
template <typename Scalar>
void TriangularSolveWrapper(hifde3d::Dense<Scalar>& R, hifde3d::Dense<Scalar>& B);

// Perform pivoted QR on A.  Overwrite upper-triangular part of A
// with R.  Fill in jpvt with pivots.
template <typename Scalar>
void PivotedQRWrapper(hifde3d::Dense<Scalar>& A, std::vector<int>& jpvt);

}
#endif  // ifndef INTERP_DECOMP_HPP_

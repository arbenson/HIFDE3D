#ifndef _ID_HPP_
#define _ID_HPP_

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
int interp_decomp(const CpxNumMat& M, CpxNumMat& W,
		  std::vector<int>& skeleton_cols,
		  std::vector<int>& redundant_cols, double epsilon);
    // TODO: should do some handling of the size of the matrix

    // QR factorization on M
    lapack_zgeqp3(M);

    // Find which indices correspond to skeleton DOFs
    skel_tol = M(0, 0) * epsilon;
    int max_ind = 0;
    for (int i = 1; i < n; ++i) {
	if (A(i, i) > skel_tol) {
	    max_ind = i;
	}
    }
    for (int i = 0; i <= max_ind; ++i) {
	skeleton_cols.push_back(perm[i]);
    }
    for (int i = max_ind + 1; i <= M.n(); ++i) {
	redundant_cols.push_back(perm[i]);
    }

    // Solve for the interpolating factor
    W.resize(skeleton_cols.size(), M.n());
    for (int i = 0; i < R.m(); ++i) {
	for (int j = 0; j < R.n(); ++j) {
	    if (j < i) {
		W(i, j) = 0;
	    } else {
		W(i, j) = M(i, j);
	    }
	}
    }
    R_skel = CpxNumMat(skeleton_cols.size(), skeleton_cols.size());
    for (int i = 0; i < R_skel.m(); ++i) {
	for (int j = 0; j < R_skel.n(); ++j) {
	    if (j < i) {
		R_skel(i, j) = 0;
	    } else {
		R_skel(i, j) = M(i, j);
	    }
	}
    }
    lapack_ztrsm(R_skel, W);
}

#endif

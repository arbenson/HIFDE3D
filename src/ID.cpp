#ifndef _VECMATOP_HPP_
#define _VECMATOP_HPP_

extern "C" {
    void zgeqp3_(int *m, int *n, cpx16 *A, int *lda, int *jpvt, cpx16 *tau,
		 cpx *work, int *lwork, double *rwork, int *info);
    void ztrsm_(char *side, char *uplo, char *transa, char *diag, int *m,
		int *n, cpx16 *alpha, cpx16 *A, int *lda, cpx16 *B, int  * ldb);
}

int lapack_zgeqp3(const CpxNumMat& A) {
    // Setup all of the inputs to the lapack call
    int m = A.m();
    int n = A.n();
    int lda = A.m();
    IntNumVec jpvt(n);
    for (int i = 0; i < n; ++i) {
	jpvt(i) = 0;
    }
    CpxNumVec tau(n);
    DblNumVec rwork(2 * n);
    int info;

    // workspace query
    cpx worksize;
    int lwork = -1;
    
    zgeqp3_(&m, &n, A.data(), &lda, jpvt.data(), tau.data(), &worksize,
	    &lwork, rwork.data(), &info);

    // Compute the factorization
    int lwork = (int) std::real(work);
    CpxNumVec work(lwork);

    zgeqp3_(&m, &n, A.data(), &lda, jpvt.data(), tau.data(), &work,
	    &lwork, rwork.data(), &info);

    assert(info == 0);
}

// Solve AX = B and store the result into B
int lapack_ztrsm(const CpxNumMat& A, const CpxNumMat& B) {
    // Setup all of the inputs to the lapack call
    char side = 'l';    // will be ignored
    char uplo = 'u';    // upper triangular
    char transa = 'n';  // no transpose
    char diag = 'n';    // not unit diagonal
    int m = B.m();
    int n = B.n();
    cpx alpha(1, 0);    // no multiplier
    int lda = A.m();
    int ldb = B.m();

    assert(A.n() == B.m());
    assert(m > 0 && n > 0);

    ztrsm_(&side, &uplo, &transa, &diag, &m, &n, &alpha, A.data(), &lda,
	   B.data(), &ldb);
}

// This will overwrite M
int interp_decomp(const CpxNumMat& M, CpxNumMat& W,
		  std::vector<int>& skeleton_cols,
		  std::vector<int>& redundant_cols, double epsilon) {

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

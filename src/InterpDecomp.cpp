#include "dmhm/core/lapack.hpp"

extern "C" {
    void strsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n,
                float *alpha, float *A, int *lda, float *B, int *ldb);
    void dtrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n,
                double *alpha, double *A, int *lda, double *B, int *ldb);
    void ctrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n,
                std::complex<float> *alpha, std::complex<float> *A, int *lda,
                std::complex<float> *B, int *ldb);
    void ztrsm_(char *side, char *uplo, char *transa, char *diag, int *m, int *n,
                std::complex<double> *alpha, std::complex<double> *A, int *lda,
                std::complex<double> *B, int *ldb);
}

// Solve RX = B, where R is upper triangular.  Overwrite B with X.
template <typename Scalar>
void TriangularSolveWrapper(Dense<Scalar>& R, Dense<Scalar>& B);

// Perform pivoted QR on A.  Overwrite upper-triangular part of A
// with R.  Fill in jpvt with pivots.
template <typename Scalar>
void PivotedQRWrapper(Dense<Scalar>& A, std::vector<int>& jpvt);

bool IsSkel(Dense<Scalar>& R, int row, double tol) {
    return std::abs(R(row, row)) > tol;
}

template <typename Scalar>
int InterpDecomp(Dense<Scalar>& M, Dense<Scalar>& W,
		 std::vector<int>& skeleton_cols,
		 std::vector<int>& redundant_cols, double epsilon) {
    std::vector<int> jpvt;
    PivotedQRWrapper(M, jpvt);

    // temporary storage of skeleton data
    std::vector<int> skel;
    std::vector<int> redundant;

    // Find which indices correspond to skeleton DOFs
    double skel_tol = std::abs(M(0, 0)) * epsilon;
    for (int i = 0; i < n; ++i) {
	if (IsSkel(R, i, skel_tol)) {
	    skeleton_cols.push_back(jpvt[i]);
	    skel.push_back(i);
	} else {
	    redundant_cols.push_back(jpvt[i]);
	    redundant.push_back(i);
	}
    }

    // Solve for the interpolating factor
    // TODO: Assuming that the diagonal of R is non-increasing, these
    //       formulations can be a bit cleaner.
    W.Resize(skel.size(), M.Width());
    for (int m = 0; m < skel.size(); ++m) {
	int i = skel[m];
	for (int j = 0; j < W.Width(); ++j) {
	    if (j < i) {
		W(i, j) = 0;
	    } else {
		W(i, j) = M(i, j);
	    }
	}
    }

    Dense<Scalar> R_skel(skel.size(), skel.size(), GENERAL);
    for (int m = 0; m < R_skel.Height(); ++m) {
	for (int n = 0; n < R_skel.Width(); ++n) {
	    int i = skel[m];
	    int j = skel[n];
	    if (j < i) {
		R_skel(i, j) = 0;
	    } else {
		R_skel(i, j) = M(i, j);
	    }
	}
    }

    TriangularSolveWrapper(R_skel, W);
}

// TODO: Move functions over to lapack.hpp

// float
void PivotedQRWrapper(int m, int n, float *A, int lda, std::vector<int>& jpvt,
		      std::vector<float>& tau) {
    const int lwork = 2 * n + (n + 1) * BLOCKSIZE;
    std::vector<float> work(lwork);
    lapack::PivotedQR(m, n, A, lda, &jpv[0], &tau[0], &work[0], &lwork);
}

// double
void PivotedQRWrapper(int m, int n, double *A, int lda, std::vector<int>& jpvt,
		      std::vector<double>& tau) {
    const int lwork = 2 * n + (n + 1) * BLOCKSIZE;
    std::vector<double> work(lwork);
    lapack::PivotedQR(m, n, A, lda, &jpv[0], &tau[0], &work[0], &lwork);
}

// complex float
void PivotedQRWrapper(int m, int n, std::complex<float> *A, int lda,
		      std::vector<int>& jpvt,
                      std::vector< std::complex<float> >& tau) {
    const int lwork = (n + 1) * BLOCKSIZE;
    std::vector< std::complex<float> > work(lwork);
    std::vector<float> rwork(lapack::PivotedQRRealWorkSize(n));
    lapack::PivotedQR(m, n, A, lda, &jpv[0], &tau[0], &work[0], &lwork, &rwork[0]);
}

// complex double
void PivotedQRWrapper(int m, int n, std::complex<double> *A, int lda,
		      std::vector<int>& jpvt,
                      std::vector< std::complex<double> >& tau) {
    const int lwork = (n + 1) * BLOCKSIZE;
    std::vector< std::complex<double> > work(lwork);
    std::vector<double> rwork(lapack::PivotedQRRealWorkSize(n));
    lapack::PivotedQR(m, n, A, lda, &jpv[0], &tau[0], &work[0], &lwork, &rwork[0]);
}

template <typename Scalar>
void PivotedQRWrapper(Dense<Scalar>& A, std::vector<int>& jpvt) {
    int m = A.Height();
    int n = A.Width();
    int lda = A.LDim();
    Scalar *buffer = A.Buffer(0, 0);
    for (int i = 0; i < n; ++i) {
	jpvt(i) = 0;
    }
    std::vector<Scalar> tau(n);
    PivotedQRWrapper(m, n, buffer, lda, jpv, tau);
}

// float
void TriangularSolve(char *side, char *uplo, char *transa, char *diag, int *m,
                     int *n, float *alpha, float *A, int *lda, float *B, int *ldb) {
    assert(m > 0 && n > 0);
    strsm_(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

// double
void TriangularSolve(char *side, char *uplo, char *transa, char *diag, int *m,
                     int *n, double *alpha, double *A, int *lda, double *B, int *ldb) {
    assert(m > 0 && n > 0);
    dtrsm_(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

// complex float
void TriangularSolve(char *side, char *uplo, char *transa, char *diag, int *m,
                     int *n, std::complex<float> *alpha, std::complex<float> *A,
                     int *lda, std::complex<float> *B, int *ldb) {
    assert(m > 0 && n > 0);
    ctrsm_(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

// complex double
void TriangularSolve(char *side, char *uplo, char *transa, char *diag, int *m,
                     int *n, std::complex<double> *alpha, std::complex<double> *A,
                     int *lda, std::complex<double> *B, int *ldb) {
    assert(m > 0 && n > 0);
    ztrsm_(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

template <typename Scalar>
void TriangularSolveWrapper(Dense<Scalar>& R, Dense<Scalar>& B) {
    // Setup all of the inputs to the lapack call
    char side = 'l';    // will be ignored
    char uplo = 'u';    // upper triangular
    char transa = 'n';  // no transpose
    char diag = 'n';    // not unit diagonal
    int m = B.Height();
    int n = B.Width();
    Scalar alpha(1.0);  // no multiplier
    int lda = R.LDim();
    int ldb = B.LDim();
    Scalar *A = R.Buffer(0, 0);
    Scalar *B = B.Buffer(0, 0);
    TriangularSolve(&side, &uplo, &transa, &diag, &m, &n, &alpha, A, &lda, B, &ldb);
}




template void PivotedQRWrapper(Dense<float>& A, std::vector<int>& jpvt);
template void PivotedQRWrapper(Dense<double>& A, std::vector<int>& jpvt);
template void PivotedQRWrapper(Dense< std::complex<float> >& A,
                               std::vector<int>& jpvt);
template void PivotedQRWrapper(Dense< std::complex<double> >& A,
                               std::vector<int>& jpvt);

template void TriangSolveWrapper(Dense<float>& R, Dense<float>& B);
template void TriangSolveWrapper(Dense<double>& R, Dense<double>& B);
template void TriangSolveWrapper(Dense< std::complex<float> >& R,
                                 Dense< std::complex<float> >& B);
template void TriangSolveWrapper(Dense< std::complex<double> >& R,
                                 Dense< std::complex<double> >& B);


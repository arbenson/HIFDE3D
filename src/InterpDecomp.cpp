#include "hifde3d.hpp"

#include "assert.h"

namespace hifde3d {

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

template <typename Scalar>
bool IsSkel(Dense<Scalar>& R, int row, double tol) {
#ifndef RELEASE
    CallStackEntry entry("IsSkel");
#endif
    return std::abs(R.Get(row, row)) > tol;
}

template <typename Scalar>
void InterpDecomp(Dense<Scalar>& M, Dense<Scalar>& W,
                 std::vector<int>& skeleton_cols,
                 std::vector<int>& redundant_cols, double epsilon) {
#ifndef RELEASE
    CallStackEntry entry("InterpDecomp");
#endif
    std::vector<int> jpvt;
    PivotedQRWrapper(M, jpvt);

    // temporary storage of skeleton data
    std::vector<int> skel;
    std::vector<int> redundant;

    // Find which indices correspond to skeleton DOFs
    double skel_tol = std::abs(M.Get(0, 0)) * epsilon;
    for (int i = 0; i < M.Width(); ++i) {
        if (IsSkel(M, i, skel_tol)) {
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
    for (size_t m = 0; m < skel.size(); ++m) {
        int i = skel[m];
        for (int j = 0; j < W.Width(); ++j) {
            if (j < i) {
                W.Set(i, j, 0);
            } else {
                W.Set(i, j, M.Get(i, j));
            }
        }
    }

    Dense<Scalar> R_skel(skel.size(), skel.size(), GENERAL);
    for (int m = 0; m < R_skel.Height(); ++m) {
        for (int n = 0; n < R_skel.Width(); ++n) {
            int i = skel[m];
            int j = skel[n];
            if (j < i) {
                R_skel.Set(i, j, 0);
            } else {
                R_skel.Set(i, j, M.Get(i, j));
            }
        }
    }

    TriangularSolveWrapper(R_skel, W);
}

// TODO: Move functions over to lapack.hpp

// float
void PivotedQRWrapper(int m, int n, float *A, int lda, std::vector<int>& jpvt,
                      std::vector<float>& tau) {
#ifndef RELEASE
    CallStackEntry entry("PivotedQRWrapper");
#endif
    int lwork = 2 * n + (n + 1) * BLOCKSIZE;
    std::vector<float> work(lwork);
    lapack::PivotedQR(m, n, A, lda, &jpvt[0], &tau[0], &work[0], lwork);
    for( size_t i=0; i<jpvt.size(); ++i )
        jpvt[i]--;
}

// double
void PivotedQRWrapper(int m, int n, double *A, int lda, std::vector<int>& jpvt,
                      std::vector<double>& tau) {
#ifndef RELEASE
    CallStackEntry entry("PivotedQRWrapper");
#endif
    int lwork = 2 * n + (n + 1) * BLOCKSIZE;
    std::vector<double> work(lwork);
    lapack::PivotedQR(m, n, A, lda, &jpvt[0], &tau[0], &work[0], lwork);
    for( size_t i=0; i<jpvt.size(); ++i )
        jpvt[i]--;
}

// complex float
void PivotedQRWrapper(int m, int n, std::complex<float> *A, int lda,
                      std::vector<int>& jpvt,
                      std::vector< std::complex<float> >& tau) {
#ifndef RELEASE
    CallStackEntry entry("PivotedQRWrapper");
#endif
    int lwork = (n + 1) * BLOCKSIZE;
    std::vector< std::complex<float> > work(lwork);
    std::vector<float> rwork(lapack::PivotedQRRealWorkSize(n));
    lapack::PivotedQR(m, n, A, lda, &jpvt[0], &tau[0], &work[0], lwork, &rwork[0]);
    for( size_t i=0; i<jpvt.size(); ++i )
        jpvt[i]--;
}

// complex double
void PivotedQRWrapper(int m, int n, std::complex<double> *A, int lda,
                      std::vector<int>& jpvt,
                      std::vector< std::complex<double> >& tau) {
#ifndef RELEASE
    CallStackEntry entry("PivotedQRWrapper");
#endif
    int lwork = (n + 1) * BLOCKSIZE;
    std::vector< std::complex<double> > work(lwork);
    std::vector<double> rwork(lapack::PivotedQRRealWorkSize(n));
    lapack::PivotedQR(m, n, A, lda, &jpvt[0], &tau[0], &work[0], lwork, &rwork[0]);
    for( size_t i=0; i<jpvt.size(); ++i )
        jpvt[i]--;
}

template <typename Scalar>
void PivotedQRWrapper(Dense<Scalar>& A, std::vector<int>& jpvt) {
#ifndef RELEASE
    CallStackEntry entry("PivotedQRWrapper");
#endif
    int m = A.Height();
    int n = A.Width();
    int lda = A.LDim();
    Scalar *buffer = A.Buffer(0, 0);
    jpvt.clear();
    jpvt.resize(n, 0);
    std::vector<Scalar> tau(n);
    PivotedQRWrapper(m, n, buffer, lda, jpvt, tau);
}

// float
void TriangularSolve(char *side, char *uplo, char *transa, char *diag, int *m,
                     int *n, float *alpha, float *A, int *lda, float *B, int *ldb) {
#ifndef RELEASE
    CallStackEntry entry("TriangularSolve");
    if (m <= 0)
	throw std::logic_error("Invalid matrix height for triangular solve");
    if (n <= 0)
	throw std::logic_error("Invalid matrix width for triangular solve");
#endif
    assert(m > 0 && n > 0);
    strsm_(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

// double
void TriangularSolve(char *side, char *uplo, char *transa, char *diag, int *m,
                     int *n, double *alpha, double *A, int *lda, double *B, int *ldb) {
#ifndef RELEASE
    CallStackEntry entry("TriangularSolve");
    if (m <= 0)
	throw std::logic_error("Invalid matrix height for triangular solve");
    if (n <= 0)
	throw std::logic_error("Invalid matrix width for triangular solve");
#endif
    assert(m > 0 && n > 0);
    dtrsm_(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

// complex float
void TriangularSolve(char *side, char *uplo, char *transa, char *diag, int *m,
                     int *n, std::complex<float> *alpha, std::complex<float> *A,
                     int *lda, std::complex<float> *B, int *ldb) {
#ifndef RELEASE
    CallStackEntry entry("TriangularSolve");
    if (m <= 0)
	throw std::logic_error("Invalid matrix height for triangular solve");
    if (n <= 0)
	throw std::logic_error("Invalid matrix width for triangular solve");
#endif
    assert(m > 0 && n > 0);
    ctrsm_(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

// complex double
void TriangularSolve(char *side, char *uplo, char *transa, char *diag, int *m,
                     int *n, std::complex<double> *alpha, std::complex<double> *A,
                     int *lda, std::complex<double> *B, int *ldb) {
#ifndef RELEASE
    CallStackEntry entry("TriangularSolve");
    if (m <= 0)
	throw std::logic_error("Invalid matrix height for triangular solve");
    if (n <= 0)
	throw std::logic_error("Invalid matrix width for triangular solve");
#endif
    assert(m > 0 && n > 0);
    ztrsm_(side, uplo, transa, diag, m, n, alpha, A, lda, B, ldb);
}

template <typename Scalar>
void TriangularSolveWrapper(Dense<Scalar>& R, Dense<Scalar>& B) {
#ifndef RELEASE
    CallStackEntry entry("TriangularSolveWrapper");
#endif
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
    Scalar *A_buf = R.Buffer(0, 0);
    Scalar *B_buf = B.Buffer(0, 0);
    TriangularSolve(&side, &uplo, &transa, &diag, &m, &n, &alpha, A_buf, &lda, B_buf, &ldb);
}


// Declarations of possible templated types
template void InterpDecomp(Dense<float>& M, Dense<float>& W,
			   std::vector<int>& skeleton_cols,
                           std::vector<int>& redundant_cols,
			   double epsilon);
template void InterpDecomp(Dense<double>& M, Dense<double>& W,
			   std::vector<int>& skeleton_cols,
                           std::vector<int>& redundant_cols,
			   double epsilon);
template void InterpDecomp(Dense< std::complex<float> >& M,
                           Dense< std::complex<float> >& W,
			   std::vector<int>& skeleton_cols,
                           std::vector<int>& redundant_cols,
			   double epsilon);
template void InterpDecomp(Dense< std::complex<double> >& M,
                           Dense< std::complex<double> >& W,
			   std::vector<int>& skeleton_cols,
                           std::vector<int>& redundant_cols,
			   double epsilon);

template void PivotedQRWrapper(Dense<float>& A, std::vector<int>& jpvt);
template void PivotedQRWrapper(Dense<double>& A, std::vector<int>& jpvt);
template void PivotedQRWrapper(Dense< std::complex<float> >& A,
                               std::vector<int>& jpvt);
template void PivotedQRWrapper(Dense< std::complex<double> >& A,
                               std::vector<int>& jpvt);
template void TriangularSolveWrapper(Dense<float>& R, Dense<float>& B);
template void TriangularSolveWrapper(Dense<double>& R, Dense<double>& B);
template void TriangularSolveWrapper(Dense< std::complex<float> >& R,
                                     Dense< std::complex<float> >& B);
template void TriangularSolveWrapper(Dense< std::complex<double> >& R,
                                     Dense< std::complex<double> >& B);

}

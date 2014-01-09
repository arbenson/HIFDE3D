#ifndef SETUP_STENCIL_HPP_
#define SETUP_STENCIL_HPP_

#include "hifde3d/core/environment.hpp"
#include "numtns.hpp"

namespace hifde3d {

template <typename Scalar>
void SetupStencil(HIFFactor<Scalar>& factor, int N, double h, NumTns<Scalar>& A,
                  NumTns<Scalar>& V) {
#ifndef RELEASE
    CallStackEntry entry("SetupStencil");
#endif
    int NC = N + 1;
    assert(A.m() == NC && A.n() == NC && A.p() == NC);
    assert(V.m() == NC && V.n() == NC && V.p() == NC);
    Sparse<Scalar>& matrix = factor.sp_matrix();
    double hh = h * h;

    for (int i = 1; i < N; ++i) {
        for (int j = 1; j < N; ++j) {
            for (int k = 1; k < N; ++k) {
                Vector<int> jidx;
                Vector<Scalar> vals;

                // Coefficient on diagonal
                Index3 ind(i, j, k);
                Scalar a = Scalar(0);
                a += (A(i - 1, j, k) + A(i + 1, j, k)) / hh;
                a += (A(i, j - 1, k) + A(i, j + 1, k)) / hh;
                a += (A(i, j, k - 1) + A(i, j, k + 1)) / hh;
                a += V(i, j, k);
                jidx.PushBack(factor.Tensor2LinearInd(ind));
                vals.PushBack(a);

                // Coefficients from neighbors
                ind = Index3(i - 1, j, k);
                jidx.PushBack(factor.Tensor2LinearInd(ind));
                vals.PushBack(A(ind) / hh);

                ind = Index3(i + 1, j, k);
                jidx.PushBack(factor.Tensor2LinearInd(ind));
                vals.PushBack(A(ind) / hh);

                ind = Index3(i, j - 1, k);
                jidx.PushBack(factor.Tensor2LinearInd(ind));
                vals.PushBack(A(ind) / hh);

                ind = Index3(i, j + 1, k);
                jidx.PushBack(factor.Tensor2LinearInd(ind));
                vals.PushBack(A(ind) / hh);

                ind = Index3(i, j, k - 1);
                jidx.PushBack(factor.Tensor2LinearInd(ind));
                vals.PushBack(A(ind) / hh);

                ind = Index3(i, j, k + 1);
                jidx.PushBack(factor.Tensor2LinearInd(ind));
                vals.PushBack(A(ind) / hh);

                ind = Index3(i, j, k);
                matrix.Add(factor.Tensor2LinearInd(ind), jidx, vals);
            }
        }
    }
}

}

#endif  // ifndef SETUP_STENCIL_HPP_

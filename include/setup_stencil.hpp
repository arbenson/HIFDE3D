#ifndef SETUP_STENCIL_HPP_
#define SETUP_STENCIL_HPP_

namespace hifde3d {

template <typename Scalar>
void SetupStencil(HIFFactor<Scalar>& factor, int N, double h, NumTns<Scalar>& A,
                  NumTns<Scalar>& V) {
#ifndef RELEASE
    CallStackEntry entry("SetupStencil");
#endif
    int NC = N + 1;
    assert(A.m() == NC + 1 && A.n() == NC + 1 && A.p() == NC + 1);
    assert(V.m() == NC + 1 && V.n() == NC + 1 && V.p() == NC + 1);
    Sparse<Scalar>& matrix = factor.sp_matrix();
    double hh = h * h;

    for (int i = 1; i <= N; ++i) {
        for (int j = 1; j <= N; ++j) {
            for (int k = 1; k <= N; ++k) {
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
                if (i - 1 >= 1) {
                    ind = Index3(i - 1, j, k);
                    jidx.PushBack(factor.Tensor2LinearInd(ind));
                    vals.PushBack(-A(ind) / hh);
                }

                if (i + 1 <= N) {
                    ind = Index3(i + 1, j, k);
                    jidx.PushBack(factor.Tensor2LinearInd(ind));
                    vals.PushBack(-A(ind) / hh);
                }

                if (j - 1 >= 1) {
                    ind = Index3(i, j - 1, k);
                    jidx.PushBack(factor.Tensor2LinearInd(ind));
                    vals.PushBack(-A(ind) / hh);
                }

                if (j + 1 <= N) {
                    ind = Index3(i, j + 1, k);
                    jidx.PushBack(factor.Tensor2LinearInd(ind));
                    vals.PushBack(-A(ind) / hh);
                }

                if (k - 1 >= 1) {
                    ind = Index3(i, j, k - 1);
                    jidx.PushBack(factor.Tensor2LinearInd(ind));
                    vals.PushBack(-A(ind) / hh);
                }

                if (k + 1 <= N) {
                    ind = Index3(i, j, k + 1);
                    jidx.PushBack(factor.Tensor2LinearInd(ind));
                    vals.PushBack(-A(ind) / hh);
                }

                ind = Index3(i, j, k);
                matrix.Add(factor.Tensor2LinearInd(ind), jidx, vals);
            }
        }
    }
}

} // namespace hifde3d

#endif  // ifndef SETUP_STENCIL_HPP_

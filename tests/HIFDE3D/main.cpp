#include "hifde3d.hpp"
#include <iostream>

using namespace hifde3d;

int main() {
#ifndef RELEASE
    // When not in release mode, we catch all errors so that we can print the
    // manual call stack.
    try {
#endif
    int N = 31;
    int P = 4;
    double epsilon = 1e-3;
    HIFFactor<double> factor(N, P, epsilon);

    int NC = factor.N() + 1;
    NumTns<double> A(NC + 1, NC + 1, NC + 1);
    NumTns<double> V(NC + 1, NC + 1, NC + 1);
    for (int i = 0; i < NC + 1; ++i) {
        for (int j = 0; j < NC + 1; ++j) {
            for (int k = 0; k < NC + 1; ++k) {
                Index3 ind(i, j, k);
                A(ind) = 1;
                if (i == 0 || j == 0 || k == 0 || i == NC
                           || j == NC || k == NC) {
                    V(ind) = 0;
                } else {
                    V(ind) = 0;
                }
            }
        }
    }

    double h = 1.0 / NC;
    std::cout << "setup..." << std::endl;
    SetupStencil(factor, NC - 1, h, A, V);
    std::cout << "setup done" << std::endl;

    factor.Initialize();
    std::cout << "factoring..." << std::endl;
    factor.Factor();
    std::cout << "factoring done" << std::endl;

    Vector<double> v_vec(NC * NC * NC);
    for (int i = 0; i < v_vec.Size(); ++i) {
	v_vec.Set(i, 1.0 / (NC * NC * NC));
    }
    Vector<double> RHS(NC * NC * NC);
    SpMV(factor.sp_matrix(), v_vec, RHS);

    Vector<double> u_vec(NC * NC * NC);
    factor.Apply(u_vec, false);
    double err_apply = RelativeErrorNorm2(RHS, u_vec);
    std::cout << "Error in application of A: " << err_apply << std::endl;

    factor.Apply(u_vec, true);
    double err_inv = RelativeErrorNorm2(v_vec, u_vec);
    std::cout << "Error in application of inverse of A: " << err_inv << std::endl;

#ifndef RELEASE
    } //end of try
    catch (std::logic_error err) {
    std::cerr << "LOGIC ERROR" << std::endl;
    std::cerr << err.what() << std::endl;
    DumpCallStack();
    }
    catch (std::runtime_error err) {
    std::cerr << "RUNTIME ERROR" << std::endl;
    std::cerr << err.what() << std::endl;
    DumpCallStack();
    }
    catch( ... ) {
    std::cerr << "Caught error." << std::endl;
    DumpCallStack();
    }
#endif
    return 0;
}


#include "hifde3d.hpp"
#include "Factor.hpp"
#include "setup_stencil.hpp"

#include <iostream>

int main() {
#ifndef RELEASE
    // When not in release mode, we catch all errors so that we can print the
    // manual call stack.
    try {
#endif
	hifde3d::HIFFactor<double> factor;
    factor.set_epsilon(1e-3);
    factor.set_N(31);
    factor.set_P(4);

    int NC = factor.N() + 1;
    hifde3d::NumTns<double> A(NC, NC, NC);
    hifde3d::NumTns<double> V(NC, NC, NC);
    hifde3d::Vector<double> u(NC * NC * NC);
    for (int i = 0; i < NC; ++i) {
	for (int j = 0; j < NC; ++j) {
	    for (int k = 0; k < NC; ++k) {
		hifde3d::Index3 ind(i, j, k);
		if (i == 0 || j == 0 || k == 0) {
		    A(ind) = 0;
		    V(ind) = 0;
		} else {
		    A(ind) = 1;
		    V(ind) = 1;
		}
		u.Set(factor.Tensor2LinearInd(ind), 1);
	    }
	}
    }

    double h = 1.0 / NC;
    std::cout << "setup..." << std::endl;
    hifde3d::SetupStencil(factor, NC - 1, h, A, V);

    factor.Initialize();
    std::cout << "factoring..." << std::endl;
    factor.Factor();
#ifndef RELEASE
    } catch( ... ) {
	std::cerr << "Caught error." << std::endl;
        DumpCallStack();
    }
#endif

    return 0;
}


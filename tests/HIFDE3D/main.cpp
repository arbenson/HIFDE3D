#include "hifde3d.hpp"
#include <iostream>

int main() {
#ifndef RELEASE
    // When not in release mode, we catch all errors so that we can print the
    // manual call stack.
    try {
#endif
	int N = 31;
	int P = 4;
	double epsilon = 1e-3;
	hifde3d::HIFFactor<double> factor(N, P, epsilon);

	int NC = factor.N() + 1;
	hifde3d::NumTns<double> A(NC + 1, NC + 1, NC + 1);
	hifde3d::NumTns<double> V(NC + 1, NC + 1, NC + 1);
	hifde3d::Vector<double> u(NC * NC * NC);
	for (int i = 0; i < NC + 1; ++i) {
	    for (int j = 0; j < NC + 1; ++j) {
		for (int k = 0; k < NC + 1; ++k) {
		    hifde3d::Index3 ind(i, j, k);
		    A(ind) = 1;
		    if (i == 0 || j == 0 || k == 0 || i == NC || j == NC || k == NC) {
			V(ind) = 0;
		    } else {
			V(ind) = 1;
		    }
		}
	    }
	}
	for (int i = 0; i < NC; ++i) {
	    for (int j = 0; j < NC; ++j) {
		for (int k = 0; k < NC; ++k) {
		    hifde3d::Index3 ind(i, j, k);
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
    std::cout << "factoring finished" << std::endl;
#ifndef RELEASE
    }
    catch (std::logic_error err) {
	std::cerr << "LOGIC ERROR" << std::endl;
	std::cerr << err.what() << std::endl;
	hifde3d::DumpCallStack();
    }
    catch (std::runtime_error err) {
	std::cerr << "RUNTIME ERROR" << std::endl;
	std::cerr << err.what() << std::endl;
	hifde3d::DumpCallStack();
    }
    catch( ... ) {
	std::cerr << "Caught error." << std::endl;
	hifde3d::DumpCallStack();
    }
#endif
    return 0;
}


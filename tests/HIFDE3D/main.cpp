#include "hifde3d.hpp"
#include <iostream>
#include <time.h>
#include <stdlib.h>

using namespace hifde3d;

int OptionsCreate(int argc, char** argv,
		  std::map<std::string, std::string>& options) {
    options.clear();
    for(int k = 1; k < argc; k = k + 2) {
      options[ std::string(argv[k]) ] = std::string(argv[k+1]);
    }
    return 0;
}

std::string FindOption(std::map<std::string, std::string>& opts,
                  std::string option) {
    std::map<std::string, std::string>::iterator mi = opts.find(option);
    if (mi == opts.end()) {
	std::cerr << "Missing option " << option << std::endl;
        return "";
    }
    return mi->second;
}

int main(int argc, char** argv) {
#ifndef RELEASE
    // When not in release mode, we catch all errors so that we can print the
    // manual call stack.
    try {
#endif
    std::map<std::string, std::string> opts;
    OptionsCreate(argc, argv, opts);
    double epsilon = 1e-3;
    std::string opt = FindOption(opts, "-tol");
    if (!opt.empty()) {
	epsilon = atof(opt.c_str());
    }
    std::cout << "Tolerance is: " << epsilon << std::endl;
    
    int N = 32-1;
    int P = 4;
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

    Vector<double> v_vec(NC * NC * NC);
    srand((unsigned)time(NULL));

    // NOTE: do the SpMV _before_ calling factor.
    // for (int i = 0; i < v_vec.Size(); ++i) {
    //	v_vec.Set(i, ((double) rand()) / (RAND_MAX+1) / NC / NC);
    // }
    for (int i = 0; i < NC; ++i) {
	for (int j = 0; j < NC; ++j) {
	    for (int k = 0; k < NC; ++k) {
		Index3 ind(i, j, k);
		if (i != 0 && j != 0 && k != 0) {
		    v_vec.Set(factor.Tensor2LinearInd(ind), 1.0);
		} else {
		    v_vec.Set(factor.Tensor2LinearInd(ind), 0.0);
		}
	    }
	}
    }

    Vector<double> w_vec(NC * NC * NC);
    // w <- Av
    SpMV(factor.sp_matrix(), v_vec, w_vec);

    factor.Initialize();
    std::cout << "factoring..." << std::endl;
    srand48(time(NULL));
    time_t t0, t1;
    t0 = time(0);
    factor.Factor();
    t1 = time(0);
    std::cout << "factoring done (" << difftime(t1, t0) << " seconds)" << std::endl;

    // u <- v
    Vector<double> u_vec(v_vec.Size());
    for (int i = 0; i < u_vec.Size(); ++i) {
	u_vec.Set(i, v_vec.Get(i));
    }
    // u <- Fu
    t0 = time(0);
    factor.Apply(u_vec, false);
    t1 = time(0);
    std::cout << "application done (" << difftime(t1, t0) << " seconds)" << std::endl;
    // ||w - u|| / ||w|| = ||Av - Fv|| // ||Av||
    double err_app = RelativeErrorNorm2(w_vec, u_vec);
    std::cout << "Relative error in application of A (e_a): " << err_app << std::endl;

    // w <- F^{-1}w = F^{-1}Av
    t0 = time(0);
    factor.Apply(w_vec, true);
    t1 = time(0);
    std::cout << "inverse application done (" << difftime(t1, t0) << " seconds)" << std::endl;
    double err_inv = RelativeErrorNorm2(v_vec, w_vec);
    std::cout << "Error in application of inverse of A (e_s): " << err_inv << std::endl;

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

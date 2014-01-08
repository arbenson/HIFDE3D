#include "Factor.hpp"

namespace hifde3d {

int main() {
    HIFFactor< std::complex<double> > factor1;
    factor1.set_epsilon(1e-3);
    factor1.set_N(32);
    factor1.set_P(4);

    HIFFactor< std::complex<float> > factor2;
    factor2.set_epsilon(1e-3);
    factor2.set_N(32);
    factor2.set_P(4);

    HIFFactor<double> factor3;
    factor3.set_epsilon(1e-3);
    factor3.set_N(32);
    factor3.set_P(4);

    HIFFactor<float> factor4;
    factor4.set_epsilon(1e-3);
    factor4.set_N(32);
    factor4.set_P(4);
}

}

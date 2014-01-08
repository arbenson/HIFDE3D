HIFDE3D
=======

Hierarchical interpolate factorization for differential equations in three dimensions.

built using:

    cd HIFDE3D
    mkdir build
    cd build
    cmake ..
    make

If something other than the reference BLAS/LAPACK libs are desired for HIFDE3D,
you should add them into the CMake command using a semi-colon delimited list
of the form:
    -DMATH_LIBS="/path/to/first/lib.a;/path/to/second/lib.a"

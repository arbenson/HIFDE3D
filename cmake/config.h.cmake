/*
   Copyright (c) 2011-2013 Austin Bensin, Yingzhou Li, Jack Poulson,
   Lexing Ying, Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (HIFDE3D) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef HIFDE3D_CONFIG_H
#define HIFDE3D_CONFIG_H 1

#define HIFDE3D_VERSION_MAJOR @HIFDE3D_VERSION_MAJOR@
#define HIFDE3D_VERSION_MINOR @HIFDE3D_VERSION_MINOR@
#cmakedefine RELEASE
#cmakedefine BLAS_POST
#cmakedefine LAPACK_POST
#cmakedefine AVOID_COMPLEX_MPI
#cmakedefine HAVE_MPI_IN_PLACE

#define RESTRICT @RESTRICT@

#endif /* ifndef HIFDE3D_CONFIG_H */

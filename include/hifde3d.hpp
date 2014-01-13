#pragma once
#ifndef HIFDE3D_HPP
#define HIFDE3D_HPP 1

#include "hifde3d/core/environment.hpp"
#include "hifde3d/core/blas.hpp"
#include "hifde3d/core/lapack.hpp"
#include "hifde3d/core/vector.hpp"
#include "hifde3d/core/dense.hpp"
#include "hifde3d/core/sparse.hpp"
#include "hifde3d/core/random.hpp"
#include "hifde3d/core/memory_map.hpp"
#include "hifde3d/core/mpi.hpp"
#include "hifde3d/core/choice.hpp"
#include "hifde3d/core/mpi_choice.hpp"
#include "hifde3d/core/hmat_tools.hpp"

//void *Sglobal;

#include "./data.hpp"
#include "./vec3t.hpp"
#include "./numtns.hpp"
#include "./Factor.hpp"
#include "./setup_stencil.hpp"
#include "./Schur.hpp"
#include "./InterpDecomp.hpp"


#endif

#include "hifde3d.hpp"
#include <iostream>

namespace hifde3d {

template <typename Scalar>
void HIFFactor<Scalar>::Initialize() {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::Initialize");
#endif
    int NC = N_ + 1;
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::Initialize");
    if (N_ <= 0)
        throw std::logic_error
              ("Number of discretization points must be positive");
    if (sp_matrix_.Height() != NC * NC * NC)
        throw std::logic_error("Sparse matrix has incorrect height");
    if (sp_matrix_.Width() != NC * NC * NC)
        throw std::logic_error("Sparse matrix has incorrect height");
#endif
    remaining_DOFs_.resize(NC, NC, NC);
    // Any zero index is zero (vanishing boundary conditions)
    for (int i = 0; i < NC; ++i) {
        for (int j = 0; j < NC; ++j) {
            for (int k = 0; k < NC; ++k) {
                if (i == 0 || j == 0 || k == 0) {
                    remaining_DOFs_(i, j, k) = 0;
                } else {
                    remaining_DOFs_(i, j, k) = 1;
                }
            }
        }
    }

    int num_levels = static_cast<int>(round(log2(NC / P_))) + 1;
    schur_level_data_.resize(num_levels);
    skel_level_data_.resize(num_levels);
}

template <typename Scalar>
void HIFFactor<Scalar>::Factor() {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::Factor");
#endif
    int NC = N_ + 1;
    int num_levels = static_cast<int>(round(log2(NC / P_))) + 1;
    std::cout << "Total Levels: " << num_levels << std::endl;

    for (int level = 0; level < num_levels; ++level) {
        //TODO: if the number on each side is not power of 2, bug in width.
        int width = pow2(level) * P_;
        int cells_per_dir = NC / width;

        std::cout << "level " << level << std::endl;
        std::cout << "Starting Schur..." << std::endl;
        LevelFactorSchur(cells_per_dir, level);
        std::cout << "Schur done" << std::endl;
        UpdateMatrixAndDOFs(level, SCHUR );

        if (level < num_levels - 1) {
            std::cout << "Starting skel..." << std::endl;
            LevelFactorSkel(cells_per_dir, level);
            std::cout << "Skel done" << std::endl;
            UpdateMatrixAndDOFs(level, SKEL );
        }
    }
}

template <typename Scalar>
Index3 HIFFactor<Scalar>::Linear2TensorInd(int ind) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::Linear2TensorInd");
#endif
    int NC = N_ + 1;
    int i = ind % NC;
    int j = ((ind - i) % (NC * NC)) / NC;
    int k = (ind - i - NC * j) / (NC * NC);
    assert(0 <= i && i <= N_ && 0 <= j && j <= N_ && 0 <= k && k <= N_);
    return Index3(i, j, k);
}

template <typename Scalar>
int HIFFactor<Scalar>::Tensor2LinearInd(Index3 ind) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::Tensor2LinearInd");
#endif
    int i = ind(0);
    int j = ind(1);
    int k = ind(2);
    int NC = N_ + 1;
    if (!(0 <= i && i <= N_ && 0 <= j && j <= N_ && 0 <= k && k <= N_)) {
    std::cout << i << " " << j << " " << k << std::endl;
    }
    assert(0 <= i && i <= N_ && 0 <= j && j <= N_ && 0 <= k && k <= N_);
    return i + NC * j + (NC * NC) * k;
}

template <typename Scalar>
void HIFFactor<Scalar>::UpdateRemainingDOFs(int level, FactorType ftype) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::UpdateRemainingDOFs");
#endif
    std::vector<int> eliminated_DOFs;

    // Gather the DOFs
    std::vector< FactorData<Scalar> > *level_data = &schur_level_data_[level];
    if (ftype == SKEL) {
        level_data = &skel_level_data_[level];
    }

    for (size_t i = 0; i < level_data->size(); ++i) {
        std::vector<int>& red_inds = (*level_data)[i].ind_data().redundant_inds();
        std::vector<int>& global_inds = (*level_data)[i].ind_data().global_inds();
        for (size_t j = 0; j < red_inds.size(); ++j) {
            eliminated_DOFs.push_back(global_inds[red_inds[j]]);
        }
    }

    // Eliminate DOFs
    for (size_t i = 0; i < eliminated_DOFs.size(); ++i) {
        Index3 ind = Linear2TensorInd(eliminated_DOFs[i]);
        remaining_DOFs_(ind(0), ind(1), ind(2)) = 0;
    }
}

template <typename Scalar>
void HIFFactor<Scalar>::SchurAfterID(FactorData<Scalar>& data) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::SchurAfterID");
#endif
    std::vector<int>& global_inds = data.ind_data().global_inds();
    Dense<Scalar> submat;
    DenseSubmatrix(sp_matrix_, global_inds, global_inds, submat);

    // Start with identity
    int size = global_inds.size();
    Dense<Scalar> Rot(size, size, GENERAL);
    for (int i = 0; i < size; ++i) {
        Rot.Set(i, i, Scalar(1));
    }

    // Fill in with W
    std::vector<int>& skel_inds = data.ind_data().skeleton_inds();
    std::vector<int>& red_inds = data.ind_data().redundant_inds();
    for (size_t i = 0; i < skel_inds.size(); ++i) {
        for (size_t j = 0; j < red_inds.size(); ++j) {
            Rot.Set(skel_inds[i], red_inds[j], -data.W_mat().Get(i, j));
        }
    }

    Dense<Scalar> tmp(submat.Height(), Rot.Width(), GENERAL);
    hmat_tools::Multiply(Scalar(1), submat, Rot, tmp);

    Dense<Scalar> result(Rot.Height(), tmp.Width(), GENERAL);
    hmat_tools::AdjointMultiply(Scalar(1), Rot, tmp, result);
    Schur(result, data);
}

template <typename Scalar>
bool HIFFactor<Scalar>::Skeletonize(Index3 cell_location, Face face,
                                    int level, FactorData<Scalar>& data) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::Skeletonize");
#endif
    SkelIndexData skel_data;
    SkelInteractionIndexData(cell_location, face, level, skel_data);
    if (skel_data.global_cols().size() == 0) {
        return false;    // No face here.
    }

    Dense<Scalar> submat;
    DenseSubmatrix
    (sp_matrix_, skel_data.global_rows(), skel_data.global_cols(), submat);
    data.set_face(face);
    std::vector<int>& cols = skel_data.global_cols();
    // TODO: avoid this copy
    std::vector<int>&  global = data.ind_data().global_inds();
    for (size_t i = 0; i < cols.size(); ++i) {
        global.push_back(cols[i]);
    }
    InterpDecomp(submat, data.W_mat(), data.ind_data().skeleton_inds(),
                 data.ind_data().redundant_inds(), epsilon_);
    SchurAfterID(data);
    return true;
}

template <typename Scalar>
void HIFFactor<Scalar>::LevelFactorSchur(int cells_per_dir, int level) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::LevelFactorSchur");
#endif
    std::vector< FactorData<Scalar> >& level_data = schur_level_data_[level];
    level_data.resize(cells_per_dir * cells_per_dir * cells_per_dir);
    int num_DOFs_eliminated = 0;
    int curr_ind = 0;
    for (int i = 0; i < cells_per_dir; ++i) {
        for (int j = 0; j < cells_per_dir; ++j) {
            for (int k = 0; k < cells_per_dir; ++k) {
                FactorData<Scalar>& factor_data = level_data[curr_ind];
                ++curr_ind;
                InteriorCellIndexData
                (Index3(i, j, k), level, factor_data.ind_data());
                if (level == 0) {
                    assert(factor_data.NumDOFsEliminated() == 27);
                }

                // Get local data from the global matrix
                std::vector<int>& global_inds =
                    factor_data.ind_data().global_inds();
                Dense<Scalar> submat;
                DenseSubmatrix(sp_matrix_, global_inds, global_inds, submat);
                Schur(submat, factor_data);
                num_DOFs_eliminated += factor_data.NumDOFsEliminated();
            }
        }
    }
    std::cout << "Level (" << level << ", Schur): "
              << num_DOFs_eliminated << " DOFs eliminated" << std::endl;
}

template <typename Scalar>
int HIFFactor<Scalar>::SkeletonizeFace(Face face, std::vector<
                                        FactorData<Scalar> >& level_data,
                                        int level,
                                        Index3 cell_location,
                                        int& curr_pos) {
    FactorData<Scalar>& data = level_data[curr_pos];
    data.set_face(face);
    bool ret = Skeletonize(cell_location, face, level, data);
    if (!ret) {
        return 0;
    }
    curr_pos++;
    return data.NumDOFsEliminated();
}


template <typename Scalar>
void HIFFactor<Scalar>::LevelFactorSkel(int cells_per_dir, int level) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::LevelFactorSkel");
#endif
    std::vector< FactorData<Scalar> >& level_data = skel_level_data_[level];
    size_t num_faces_total = 3 * (cells_per_dir - 1) * cells_per_dir * cells_per_dir;
    level_data.resize(num_faces_total);
    int num_DOFs_eliminated = 0;
    int curr_pos = 0;
    for (int i = 0; i < cells_per_dir; ++i) {
        for (int j = 0; j < cells_per_dir; ++j) {
            for (int k = 0; k < cells_per_dir; ++k) {
                Index3 cell_location(i, j, k);
                // We only do half of the faces since each face is shared by
                // two cells.  To keep indexing consistent, we do the top,
                // front, and left faces.
                num_DOFs_eliminated += SkeletonizeFace(TOP, level_data, level,
                                                       cell_location, curr_pos);
                num_DOFs_eliminated += SkeletonizeFace(FRONT, level_data, level,
                                                       cell_location, curr_pos);
                num_DOFs_eliminated += SkeletonizeFace(LEFT, level_data, level,
                                                       cell_location, curr_pos);
            }
        }
    }
    std::cout << "Level (" << level << ", Skel): "
              << num_DOFs_eliminated << " DOFs eliminated" << std::endl;
}

template <typename Scalar>
void HIFFactor<Scalar>::UpdateMatrixAndDOFs(int level, FactorType ftype) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::UpdateMatrixAndDOFs");
#endif
    std::vector< FactorData<Scalar> > *level_data = &schur_level_data_[level];
    if (ftype == SKEL) {
        level_data = &skel_level_data_[level];
    }

    std::map<int, std::pair< Vector<int>, Vector<Scalar> > > vals;
    Vector<int> del_inds;

    // TODO: this process of forming iidx, jidx, and vals could be faster.
    std::cout << "Number of FactorDatas to process: " << level_data->size()
              << std::endl;
    for (size_t n = 0; n < level_data->size(); ++n) {
        FactorData<Scalar>& data = (*level_data)[n];
        IndexData& ind_data = data.ind_data();
        std::vector<int>& skel_inds = ind_data.skeleton_inds();
        std::vector<int>& global_inds = ind_data.global_inds();
        Dense<Scalar>& S = data.Schur_comp();
        assert(S.Height() == S.Width());
        assert(S.Height() == static_cast<int>(skel_inds.size()));

        for (size_t i = 0; i < skel_inds.size(); ++i) {
            for (size_t j = 0; j < skel_inds.size(); ++j) {
                assert(skel_inds[i]<global_inds.size());
                assert(skel_inds[j]<global_inds.size());

                vals[global_inds[skel_inds[i]]].first.PushBack
                (global_inds[skel_inds[j]]);
                vals[global_inds[skel_inds[i]]].second.PushBack(S.Get(i, j));
            }
        }
        // save on storage
        S.Clear();
        std::vector<int>& red_inds = ind_data.redundant_inds();
        for (size_t i = 0; i < red_inds.size(); ++i) {
                assert(red_inds[i]<global_inds.size());
            del_inds.PushBack(global_inds[red_inds[i]]);
        }
    }

    std::cout << "deleting rows..." << std::endl;
    sp_matrix_.DeleteRow(del_inds);
    std::cout << "deleting columns..." << std::endl;
    sp_matrix_.DeleteCol(del_inds);
    std::cout << "updating matrix..." << std::endl;
    for (typename std::map<int, std::pair< Vector<int>, Vector<Scalar> > >::iterator it = vals.begin();
     it != vals.end(); ++it) {
        sp_matrix_.Add(it->first, it->second.first, it->second.second);
    }
    UpdateRemainingDOFs(level, ftype);
}

template <typename Scalar>
bool HIFFactor<Scalar>::IsInterior(int level, int a) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::IsInterior");
#endif
    int width = pow2(level) * P_;
    return (a > 0 && a <= N_ && (a % width) != 0);
}


template <typename Scalar>
bool HIFFactor<Scalar>::IsFaceInterior(int level, Index3 ind) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::IsFaceInterior");
#endif
    int a_int = IsInterior(level, ind(0));
    int b_int = IsInterior(level, ind(1));
    int c_int = IsInterior(level, ind(2));
    return ((a_int && b_int && !c_int) ||
            (a_int && !b_int && c_int) ||
            (!a_int && b_int && c_int));
}

template <typename Scalar>
bool HIFFactor<Scalar>::IsEdgeInterior(int level, Index3 ind) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::IsFaceInterior");
#endif
    int a_int = IsInterior(level, ind(0));
    int b_int = IsInterior(level, ind(1));
    int c_int = IsInterior(level, ind(2));
    return ((a_int && !b_int && !c_int) ||
            (!a_int && b_int && !c_int) ||
            (!a_int && !b_int && c_int));
}

template <typename Scalar>
bool HIFFactor<Scalar>::IsCellInterior(int level, Index3 ind) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::IsCellInterior");
#endif
    return IsInterior(level, ind(0)) &&
           IsInterior(level, ind(1)) &&
           IsInterior(level, ind(2));
}

template <typename Scalar>
void HIFFactor<Scalar>::InteriorCellIndexData(Index3 cell_location, int level,
                                              IndexData& data) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::InteriorCellIndexData");
#endif
    int width = pow2(level) * P_;
    Index3 min_inds = vec3max(width * cell_location, 1);
    Index3 max_inds = vec3min(width * (cell_location + 1), N_);
    assert(min_inds <= max_inds);

    std::vector<int>& global_inds = data.global_inds();
    std::vector<int>& red_inds = data.redundant_inds();
    std::vector<int>& skel_inds = data.skeleton_inds();
    int curr_lin_index = 0;
    for (int i = min_inds(0); i <= max_inds(0); ++i) {
        for (int j = min_inds(1); j <= max_inds(1); ++j) {
            for (int k = min_inds(2); k <= max_inds(2); ++k) {
                Index3 curr_ind(i, j, k);
                if (IsRemainingDOF(curr_ind)) {
                    if (IsCellInterior(level, curr_ind)) {
                        global_inds.push_back(Tensor2LinearInd(curr_ind));
                        red_inds.push_back(curr_lin_index);
                        ++curr_lin_index;
                    }  else if (IsFaceInterior(level, curr_ind)) {
                        global_inds.push_back(Tensor2LinearInd(curr_ind));
                        skel_inds.push_back(curr_lin_index);
                        ++curr_lin_index;
                    }
                }
            }
        }
    }
    //std::cout << skel_inds.size() << std::endl;
}

template <typename Scalar>
bool HIFFactor<Scalar>::IsRemainingDOF(Index3 ind) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::IsRemainingDOF");
#endif
    return remaining_DOFs_(ind(0), ind(1), ind(2));
}

template <typename Scalar>
void HIFFactor<Scalar>::InteriorFaceDOFs(Index3 cell_location, Face face,
                                         int level, std::vector<int>& face_inds) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::InteriorFaceDOFs");
#endif
    int width = pow2(level) * P_;
    Index3 inds = cell_location * width;
    Index3 min_inds = vec3max(width * cell_location, 1);
    Index3 max_inds = vec3min(width * (cell_location + 1), N_);
    assert(min_inds <= max_inds);
    // TODO: abstract away some of the common pieces of code in the switch statement
    switch (face) {
    case TOP:
        {
            int i = min_inds(0);
            if (i == 1) {
                return;  // No top face
            }
            for (int j = min_inds(1); j <= max_inds(1); ++j) {
                for (int k = min_inds(2); k <= max_inds(2); ++k) {
                    Index3 curr_ind = Index3(i, j, k);
                    if (IsRemainingDOF(curr_ind) && IsFaceInterior(level, curr_ind)) {
                        face_inds.push_back(Tensor2LinearInd(curr_ind));
                    }
                }
            }
        }
        return;

    case BOTTOM:
        {
            int i = max_inds(0);
            if (i == N_) {
                return;  // No bottom face
            }
            for (int j = min_inds(1); j <= max_inds(1); ++j) {
                for (int k = min_inds(2); k <= max_inds(2); ++k) {
                    Index3 curr_ind = Index3(i, j, k);
                    if (IsRemainingDOF(curr_ind) && IsFaceInterior(level, curr_ind)) {
                        face_inds.push_back(Tensor2LinearInd(curr_ind));
                    }
                }
            }
        }
        return;

    case LEFT:
        {
            int j = min_inds(1);
            if (j == 1) {
                return;  // No left face
            }
            for (int i = min_inds(0); i <= max_inds(0); ++i) {
                for (int k = min_inds(2); k <= max_inds(2); ++k) {
                    Index3 curr_ind = Index3(i, j, k);
                    if (IsRemainingDOF(curr_ind) && IsFaceInterior(level, curr_ind)) {
                        face_inds.push_back(Tensor2LinearInd(curr_ind));
                    }
                }
            }
        }
        return;

    case RIGHT:
        {
            int j = max_inds(1);
            if (j == N_) {
                return;  // No right face
            }
            for (int i = min_inds(0); i <= max_inds(0); ++i) {
                for (int k = min_inds(2); k <= max_inds(2); ++k) {
                    Index3 curr_ind = Index3(i, j, k);
                    if (IsRemainingDOF(curr_ind) && IsFaceInterior(level, curr_ind)) {
                        face_inds.push_back(Tensor2LinearInd(curr_ind));
                    }
                }
            }
        }
        return;

    case FRONT:
        {
            int k = min_inds(2);
            if (k == 0) {
                return;  // No front face
            }
            for (int i = min_inds(0); i <= max_inds(0); ++i) {
                for (int j = min_inds(1); j <= max_inds(1); ++j) {
                    Index3 curr_ind = Index3(i, j, k);
                    if (IsRemainingDOF(curr_ind) && IsFaceInterior(level, curr_ind)) {
                        face_inds.push_back(Tensor2LinearInd(curr_ind));
                    }
                }
            }
        }
        return;

    case BACK:
        {
            int k = max_inds(2);
            if (k == N_) {
                return;  // No back face
            }
            for (int i = min_inds(0); i <= max_inds(0); ++i) {
                for (int j = min_inds(1); j <= max_inds(1); ++j) {
                    Index3 curr_ind = Index3(i, j, k);
                    if (IsRemainingDOF(curr_ind) && IsFaceInterior(level, curr_ind)) {
                        face_inds.push_back(Tensor2LinearInd(curr_ind));
                    }
                }
            }
        }
        return;

    default:
        // TODO: better error
        assert(0);  // not a valid face
    }
}

template <typename Scalar>
void HIFFactor<Scalar>::InteriorEdgeDOFs(Index3 cell_location, Face face,
					 int level, std::vector<int>& edge_inds) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::InteriorEdgeDOFs");
#endif
    // TODO: this routine is doing a bunch of redundant computation.  It takes
    //       code from InteriorFaceDOFs and just checks if it is an edge
    //       index (instead of a face index).
    int width = pow2(level) * P_;
    Index3 inds = cell_location * width;
    Index3 min_inds = vec3max(width * cell_location, 1);
    Index3 max_inds = vec3min(width * (cell_location + 1), N_);
    assert(min_inds <= max_inds);
    // TODO: abstract away some of the common pieces of code in the switch statement
    switch (face) {
    case TOP:
        {
            int i = min_inds(0);
            if (i == 1) {
                return;  // No top face
            }
            for (int j = min_inds(1); j <= max_inds(1); ++j) {
                for (int k = min_inds(2); k <= max_inds(2); ++k) {
                    Index3 curr_ind = Index3(i, j, k);
                    if (IsRemainingDOF(curr_ind) && IsEdgeInterior(level, curr_ind)) {
                        edge_inds.push_back(Tensor2LinearInd(curr_ind));
                    }
                }
            }
        }
        return;

    case BOTTOM:
        {
            int i = max_inds(0);
            if (i == N_) {
                return;  // No bottom face
            }
            for (int j = min_inds(1); j <= max_inds(1); ++j) {
                for (int k = min_inds(2); k <= max_inds(2); ++k) {
                    Index3 curr_ind = Index3(i, j, k);
                    if (IsRemainingDOF(curr_ind) && IsEdgeInterior(level, curr_ind)) {
                        edge_inds.push_back(Tensor2LinearInd(curr_ind));
                    }
                }
            }
        }
        return;

    case LEFT:
        {
            int j = min_inds(1);
            if (j == 1) {
                return;  // No left face
            }
            for (int i = min_inds(0); i <= max_inds(0); ++i) {
                for (int k = min_inds(2); k <= max_inds(2); ++k) {
                    Index3 curr_ind = Index3(i, j, k);
                    if (IsRemainingDOF(curr_ind) && IsEdgeInterior(level, curr_ind)) {
                        edge_inds.push_back(Tensor2LinearInd(curr_ind));
                    }
                }
            }
        }
        return;

    case RIGHT:
        {
            int j = max_inds(1);
            if (j == N_) {
                return;  // No right face
            }
            for (int i = min_inds(0); i <= max_inds(0); ++i) {
                for (int k = min_inds(2); k <= max_inds(2); ++k) {
                    Index3 curr_ind = Index3(i, j, k);
                    if (IsRemainingDOF(curr_ind) && IsEdgeInterior(level, curr_ind)) {
                        edge_inds.push_back(Tensor2LinearInd(curr_ind));
                    }
                }
            }
        }
        return;

    case FRONT:
        {
            int k = min_inds(2);
            if (k == 0) {
                return;  // No front face
            }
            for (int i = min_inds(0); i <= max_inds(0); ++i) {
                for (int j = min_inds(1); j <= max_inds(1); ++j) {
                    Index3 curr_ind = Index3(i, j, k);
                    if (IsRemainingDOF(curr_ind) && IsEdgeInterior(level, curr_ind)) {
                        edge_inds.push_back(Tensor2LinearInd(curr_ind));
                    }
                }
            }
        }
        return;

    case BACK:
        {
            int k = max_inds(2);
            if (k == N_) {
                return;  // No back face
            }
            for (int i = min_inds(0); i <= max_inds(0); ++i) {
                for (int j = min_inds(1); j <= max_inds(1); ++j) {
                    Index3 curr_ind = Index3(i, j, k);
                    if (IsRemainingDOF(curr_ind) && IsEdgeInterior(level, curr_ind)) {
                        edge_inds.push_back(Tensor2LinearInd(curr_ind));
                    }
                }
            }
        }
        return;
    default:
	assert(0);  // No face
    }
}

template <typename Scalar>
void HIFFactor<Scalar>::SkelInteractionIndexData(Index3 cell_location, Face face,
                                              int level, SkelIndexData& data) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::SkelInteractionIndexData");
#endif
    // TODO: When we loop over all cells, there is redundant computation.
    //       Here, we just compute what we need for the given cell.
    int width = pow2(level) * P_;
    Index3 min_inds = vec3max(width * cell_location, 1);
    std::vector<int>& global_rows = data.global_rows();
    std::vector<int>& global_cols = data.global_cols();

    // TODO: abstract away some of the common pieces of code in the switch statement
    switch (face) {
    case TOP:
        if (min_inds(0) == 1) {
            return;  // no top face
        }
        // Columns correspond to the face we are dealing with
        InteriorFaceDOFs(cell_location, TOP, level, global_cols);

        // Rows are all possible neighbors of interior DOFs
        // First, all interior faces for current box, except TOP
        // DO NOT INCLUDE SELF-INTERACTION
        InteriorFaceDOFs(cell_location, BOTTOM, level, global_rows);
        InteriorFaceDOFs(cell_location, LEFT, level, global_rows);
        InteriorFaceDOFs(cell_location, RIGHT, level, global_rows);
        InteriorFaceDOFs(cell_location, FRONT, level, global_rows);
        InteriorFaceDOFs(cell_location, BACK, level, global_rows);

        // Now, get all in neighbor cell.
        cell_location(0) -= 1;
        InteriorFaceDOFs(cell_location, TOP, level, global_rows);
        // BOTTOM has already been counted
        InteriorFaceDOFs(cell_location, LEFT, level, global_rows);
        InteriorFaceDOFs(cell_location, RIGHT, level, global_rows);
        InteriorFaceDOFs(cell_location, FRONT, level, global_rows);
        InteriorFaceDOFs(cell_location, BACK, level, global_rows);

	// Get the edges for this face
	InteriorEdgeDOFs(cell_location, TOP, level, global_rows);
	
    break;

    case LEFT:
        if (min_inds(1) == 1) {
            return;  // no left face
        }
        // Columns correspond to the face we are dealing with
        InteriorFaceDOFs(cell_location, LEFT, level, global_cols);

        // Rows are all possible neighbors of interior DOFs
        // First, all interior faces for current box
        InteriorFaceDOFs(cell_location, TOP, level, global_rows);
        InteriorFaceDOFs(cell_location, BOTTOM, level, global_rows);
        // DO NOT INCLUDE SELF-INTERACTION
        InteriorFaceDOFs(cell_location, RIGHT, level, global_rows);
        InteriorFaceDOFs(cell_location, FRONT, level, global_rows);
        InteriorFaceDOFs(cell_location, BACK, level, global_rows);

        // Now, get all in neighbor cell.
        cell_location(1) -= 1;
        InteriorFaceDOFs(cell_location, TOP, level, global_rows);
        InteriorFaceDOFs(cell_location, BOTTOM, level, global_rows);
        InteriorFaceDOFs(cell_location, LEFT, level, global_rows);
        // RIGHT has already been counted
        InteriorFaceDOFs(cell_location, FRONT, level, global_rows);
        InteriorFaceDOFs(cell_location, BACK, level, global_rows);

	// Get the edges for this face
	InteriorEdgeDOFs(cell_location, LEFT, level, global_rows);
    break;

    case FRONT:
        if (min_inds(2) == 1) {
            return;  // no front face
        }
        // Columns correspond to the face we are dealing with
        InteriorFaceDOFs(cell_location, FRONT, level, global_cols);

        // Rows are all possible neighbors of interior DOFs
        // First, all interior faces for current box
        InteriorFaceDOFs(cell_location, TOP, level, global_rows);
        InteriorFaceDOFs(cell_location, BOTTOM, level, global_rows);
        InteriorFaceDOFs(cell_location, LEFT, level, global_rows);
        InteriorFaceDOFs(cell_location, RIGHT, level, global_rows);
        // DO NOT INCLUDE SELF-INTERACTION
        InteriorFaceDOFs(cell_location, BACK, level, global_rows);

        // Now, get all in neighbor cell.
        cell_location(2) -= 1;
        InteriorFaceDOFs(cell_location, TOP, level, global_rows);
        InteriorFaceDOFs(cell_location, BOTTOM, level, global_rows);
        InteriorFaceDOFs(cell_location, LEFT, level, global_rows);
        InteriorFaceDOFs(cell_location, RIGHT, level, global_rows);
        InteriorFaceDOFs(cell_location, FRONT, level, global_rows);
        // BACK has already been counted

	// Get the edges for this face
	InteriorEdgeDOFs(cell_location, FRONT, level, global_rows);
    break;

    case BOTTOM:
    case RIGHT:
    case BACK:
    default:
        // TODO: better error message
        assert(0);  // not yet implemented
    }
    return;
}

template <typename Scalar>
void HIFFactor<Scalar>::set_N(int N) {
    assert(N > 0);
    N_ = N;
}

template <typename Scalar>
int HIFFactor<Scalar>::N() { return N_; }

template <typename Scalar>
void HIFFactor<Scalar>::set_P(int P) {
    assert(P > 0);
    P_ = P;
}

template <typename Scalar>
int HIFFactor<Scalar>::P() { return P_; }

template <typename Scalar>
void HIFFactor<Scalar>::set_epsilon(double epsilon) {
    assert(epsilon > 0);
    epsilon_ = epsilon;
}

template <typename Scalar>
double HIFFactor<Scalar>::epsilon() { return epsilon_; }

template <typename Scalar>
Sparse<Scalar>& HIFFactor<Scalar>::sp_matrix() { return sp_matrix_; }


// CODE TO DO APPLICATION

// Obtain u_skel = u(skel), where skel is the set of skeleton DOFs.
// Here, 'skeleton DOFs' refers to DOFs _not_ being eliminated.
// The skeleton points could be the interior of faces _or_ the skeleton
// points from skeletonization.
//
// u (in): vector of size N^3
// data (in): factor data that contains the skeleton indices
// skel_vec (out): u at the skeleton points
template <typename Scalar>
void GetSkeletonVector(Vector<Scalar>& u, FactorData<Scalar>& data,
                       Vector<Scalar>& skel_vec) {
#ifndef RELEASE
    CallStackEntry entry("GetSkeletonVector");
#endif
    std::vector<int>& global_inds = data.ind_data().global_inds();
    std::vector<int>& skel_inds = data.ind_data().skeleton_inds();
    skel_vec.Resize(skel_inds.size());
    for (size_t i = 0; i < skel_inds.size(); ++i) {
        assert(skel_inds[i] < global_inds.size());
        assert(global_inds[skel_inds[i]] < u.Size());
        skel_vec.Set(i, u.Get(global_inds[skel_inds[i]]));
    }
}

// Obtain u_red = u(red), where red is the set of redundant DOFs.
//
// u (in): vector of size N^3
// data (in): factor data that contains the redundant indices
// skel_vec (out): u at the skeleton points
template <typename Scalar>
void GetRedundantVector(Vector<Scalar>& u, FactorData<Scalar>& data,
                        Vector<Scalar>& red_vec) {
#ifndef RELEASE
    CallStackEntry entry("GetRedundantVector");
#endif
    std::vector<int>& global_inds = data.ind_data().global_inds();
    std::vector<int>& red_inds = data.ind_data().redundant_inds();
    red_vec.Resize(red_inds.size());
    for (size_t i = 0; i < red_inds.size(); ++i) {
        assert(red_inds[i] < global_inds.size());
        assert(global_inds[red_inds[i]] < u.Size());
        red_vec.Set(i, u.Get(global_inds[red_inds[i]]));
    }
}

// Copy the skeleton vector back into the global vector, u.
//
// u (out): vector of size N^3 that gets updated
// data (in): factor data containing skeleton indices
// skel_vec (in): u at the skeleton points, after updating
template <typename Scalar>
void CopySkeletonVector(Vector<Scalar>& u, FactorData<Scalar>& data,
                        Vector<Scalar>& skel_vec) {
#ifndef RELEASE
    CallStackEntry entry("CopySkeletonVector");
#endif
    std::vector<int>& global_inds = data.ind_data().global_inds();
    std::vector<int>& skel_inds = data.ind_data().skeleton_inds();
    assert(skel_vec.Size() == skel_inds.size());
    for (size_t i = 0; i < skel_inds.size(); ++i) {
        assert(skel_inds[i] < global_inds.size());
        u.Set(global_inds[skel_inds[i]], skel_vec[i]);
    }
}

// Copy the redundant vector back into the global vector, u.
//
// u (out): vector of size N^3 that gets updated
// data (in): factor data containing redundant indices
// red_vec (in): u at the redundant points, after updating
template <typename Scalar>
void CopyRedundantVector(Vector<Scalar>& u, FactorData<Scalar>& data,
                         Vector<Scalar>& red_vec) {
#ifndef RELEASE
    CallStackEntry entry("CopyRedundantVector");
#endif
    std::vector<int>& global_inds = data.ind_data().global_inds();
    std::vector<int>& red_inds = data.ind_data().redundant_inds();
    assert(red_vec.Size() == red_inds.size());
    for (size_t i = 0; i < red_inds.size(); ++i) {
        assert(red_inds[i] < global_inds.size());
        u.Set(global_inds[red_inds[i]], red_vec[i]);
    }
}

// u_skel := alpha op(A) * u_red + u_skel
//
// op(A) is either A or A^H, dictated by the input 'adjoint'
// alpha is plus or minus 1, dictated by the input 'negative'.
// A is either data.W_mat() (default) or data.X_mat() (dictated by the input
// 'is_X' set to True)
// This function updates the entries of u at the skeleton points.
//
// u (in): vector or right-hand-side
// data (in): factorization data that stores op(A)
// u_skel (in): u at the skeleton points, i.e., the DOFs remaining.
//              This is not necessarily due to skeletonization.  u_skel
//              could correspond to the interior of faces in the Schur
//              factorization _or_ the skeleton points in skeletonization.
// u_red (in): u at the redundant points, i.e., the DOFs being eliminated.
// adjoint (in): if true, op(A) = A^H; otherwise, op(A) = A
// negative (in): if true, alpha = -1; otherwise, alpha = 1
// is_X (in): if true, A = data.X_mat(); otherwise, A = data.W_mat()
template <typename Scalar>
void UpdateSkeleton(Vector<Scalar>& u, FactorData<Scalar>& data,
                    Vector<Scalar>& u_skel, Vector<Scalar>& u_red,
                    bool adjoint, bool negative, bool is_X) {
#ifndef RELEASE
    CallStackEntry entry("UpdateSkeleton");
#endif
    if (u_skel.Size() == 0 || u_red.Size() == 0) {
	return;  // nothing to do
    }
    Scalar alpha = Scalar(1.0);
    if (negative) {
        alpha = Scalar(-1.0);
    }
    Dense<Scalar>& A = data.W_mat();
    if (is_X) {
        A = data.X_mat();
    }
    if (adjoint) {
        hmat_tools::AdjointMultiply(alpha, A, u_red, Scalar(1.0), u_skel);
    } else {
        hmat_tools::Multiply(alpha, A, u_red, Scalar(1.0), u_skel);
    }
    CopySkeletonVector(u, data, u_skel);
}

// u_red := alpha op(A) * u_skel + u_red
//
// op(A) is either A or A^H, dictated by the input 'adjoint'
// alpha is plus or minus 1, dictated by the input 'negative'.
// A is either data.W_mat() (default) or data.X_mat() (dictated by the input
// 'is_X' set to True)
// This function updates the entries of u at the redundant points.
//
// u (in): vector or right-hand-side
// data (in): factorization data that stores op(A)
// u_skel (in): u at the skeleton points, i.e., the DOFs remaining.
//              This is not necessarily due to skeletonization.  u_skel
//              could correspond to the interior of faces in the Schur
//              factorization _or_ the skeleton points in skeletonization.
// u_red (in): u at the redundant points, i.e., the DOFs being eliminated.
// adjoint (in): if true, op(A) = A^H; otherwise, op(A) = A
// negative (in): if true, alpha = -1; otherwise, alpha = 1
// is_X (in): if true, A = data.X_mat(); otherwise, A = data.W_mat()
template <typename Scalar>
void UpdateRedundant(Vector<Scalar>& u, FactorData<Scalar>& data,
                     Vector<Scalar>& u_skel, Vector<Scalar>& u_red,
                     bool adjoint, bool negative, bool is_X) {
#ifndef RELEASE
    CallStackEntry entry("UpdateRedundant");
#endif
    if (u_skel.Size() == 0 || u_red.Size() == 0) {
	return;  // nothing to do
    }

    Scalar alpha = Scalar(1.0);
    if (negative) {
        alpha = Scalar(-1.0);
    }
    Dense<Scalar>& A = data.W_mat();
    if (is_X) {
        A = data.X_mat();
    }
    if (adjoint) {
        hmat_tools::AdjointMultiply(alpha, A, u_skel, Scalar(1.0), u_red);
    } else {
        hmat_tools::Multiply(alpha, A, u_skel, Scalar(1.0), u_red);
    }
    CopyRedundantVector(u, data, u_red);
}

// Apply A_{22} or A_{22}^{-1} to the redundant DOFs.
//
// u (out): vector that gets updated
// data (in): factorization data that contains the redundant indices, A_{22}, and A_{22}^{-1}
// inverse (in): if true, applies A_{22}^{-1}; otherwise, applies A_{22}
template <typename Scalar>
void ApplyA22(Vector<Scalar>& u, FactorData<Scalar>& data, bool inverse, int level) {
#ifndef RELEASE
    CallStackEntry entry("ApplyA22");
#endif
    Vector<Scalar> u_red;
    Vector<Scalar> result;
    GetRedundantVector(u, data, u_red);
    if (u_red.Size() == 0) {
	return;
    }
    result.Resize(u_red.Size());
    if (inverse) {
        hmat_tools::Multiply(Scalar(1.0), data.A_22_inv(), u_red, result);
    } else {
        hmat_tools::Multiply(Scalar(1.0), data.A_22(), u_red, result);
    }
    CopyRedundantVector(u, data, result);
}

template <typename Scalar>
void HIFFactor<Scalar>::Apply(Vector<Scalar>& u, bool apply_inverse) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::Apply");
#endif
    int num_levels = schur_level_data_.size();
    assert(num_levels == static_cast<int>(skel_level_data_.size()));
    std::cout << "Number of levels: " << num_levels << std::endl;
    assert(u.Size() == (N_ + 1) * (N_ + 1) * (N_ + 1));

    Vector<Scalar> u_skel;
    Vector<Scalar> u_red;

    std::cout << "First pass..." << std::endl;

    for (int level = 0; level < num_levels - 1; ++level) {
	std::cout << "Applying at level " << level << std::endl;
	std::cout << "Schur" << std::endl;
        for (size_t j = 0; j < schur_level_data_[level].size(); ++j) {
            FactorData<Scalar>& data = schur_level_data_[level][j];
            GetSkeletonVector(u, data, u_skel);
            GetRedundantVector(u, data, u_red);
            if (apply_inverse) {
                // u_skel := - X^H * u_red + u_skel
                UpdateSkeleton(u, data, u_skel, u_red, true, true, true);
            } else {
                // u_red := X * u_skel + u_red
                UpdateRedundant(u, data, u_skel, u_red, false, false, true);
            }
        }
	std::cout << "Skeleton" << std::endl;
        for (size_t j = 0; j < skel_level_data_[level].size(); ++j) {
            FactorData<Scalar>& data = skel_level_data_[level][j];
            GetSkeletonVector(u, data, u_skel);
            GetRedundantVector(u, data, u_red);
            if (apply_inverse) {
                // u_red := -W^H * u_skel + u_red
                UpdateRedundant(u, data, u_skel, u_red, true, true, false);
                // u_skel := -X^H * u_red + u_skel
                UpdateSkeleton(u, data, u_skel, u_red, true, true, true);
            } else {
                // u_skel := W * u_red + u_skel
                UpdateSkeleton(u, data, u_skel, u_red, false, false, false);
                // u_red := X * u_skel + u_red
                UpdateRedundant(u, data, u_skel, u_red, false, false, true);
            }
        }
    }

    std::cout << "Applying inverses..." << std::endl;
    std::cout << schur_level_data_[0].size() << std::endl;

    for (int level = 0; level < num_levels; ++level) {
	std::cout << "Applying at level " << level << std::endl;
        for (size_t j = 0; j < schur_level_data_[level].size(); ++j) {
            FactorData<Scalar>& data = schur_level_data_[level][j];
            // u_red := A^{-1}_22 * u_red
            ApplyA22(u, data, apply_inverse, level);
        }
        for (size_t j = 0; j < skel_level_data_[level].size(); ++j) {
            FactorData<Scalar>& data = skel_level_data_[level][j];
            // u_red := A_22 * u_red
            ApplyA22(u, data, apply_inverse, level);
        }
    }

    std::cout << "Second pass..." << std::endl;

    for (int level = num_levels - 2; level >= 0; --level) {
        for (size_t j = 0; j < skel_level_data_[level].size(); ++j) {
            FactorData<Scalar>& data = skel_level_data_[level][j];
            GetSkeletonVector(u, data, u_skel);
            GetRedundantVector(u, data, u_red);
            if (apply_inverse) {
                // u_red := -X * u_skel + u_red
                UpdateRedundant(u, data, u_skel, u_red, false, true, true);
                // u_skel := -W * u_red + u_skel
                UpdateSkeleton(u, data, u_skel, u_red, false, true, false);
            } else {
                // u_skel := X * u_red + u_skel
                UpdateSkeleton(u, data, u_skel, u_red, true, false, true);
                // u_red := W * u_skel + u_red
                UpdateRedundant(u, data, u_skel, u_red, true, false, false);
            }
        }
        for (size_t j = 0; j < schur_level_data_[level].size(); ++j) {
            FactorData<Scalar>& data = schur_level_data_[level][j];
            GetSkeletonVector(u, data, u_skel);
            GetRedundantVector(u, data, u_red);
            if (apply_inverse) {
                // u_red := -X * u_skel + u_red
                UpdateRedundant(u, data, u_skel, u_red, false, true, false);
            } else {
                // u_skel := X * u_red + u_skel
                UpdateSkeleton(u, data, u_skel, u_red, true, false, true);
            }
        }
    }
}

template <typename Scalar>
double RelativeErrorNorm2(Vector<Scalar>& x, Vector<Scalar>& y) {
    assert(x.Size() == y.Size());
    double err = 0;
    double norm = 0;
    for (int i = 0; i < x.Size(); ++i) {
        Scalar xi = x.Get(i);
        Scalar yi = y.Get(i);
        double diff = std::abs(xi - yi);
        err += diff * diff;
        norm += std::abs(xi) * std::abs(xi);
    }
    std::cout << "norm x: " << norm << std::endl;
    std::cout << "err: " << err << std::endl;
    return sqrt(err / norm);
}

template <typename Scalar>
void SpMV(Sparse<Scalar>& A, Vector<Scalar>& x, Vector<Scalar>& y) {
    assert(x.Size() == y.Size());
    for (int i = 0; i < x.Size(); ++i) {
        Vector<Scalar> row;
        Vector<int> inds;
        A.FindCol(i, row, inds);
        Scalar val = Scalar(0);
        assert(row.Size() == inds.Size());
        for (int j = 0; j < row.Size(); ++j) {
            val += row.Get(j) * x.Get(inds.Get(j));
        }
        y.Set(i, val);
    }
}



// Declarations
template class HIFFactor<float>;
template class HIFFactor<double>;
template class HIFFactor< std::complex<float> >;
template class HIFFactor< std::complex<double> >;

template double RelativeErrorNorm2(Vector<float>& x, Vector<float>& y);
template double RelativeErrorNorm2(Vector<double>& x, Vector<double>& y);
template double RelativeErrorNorm2(Vector< std::complex<float> >& x, Vector< std::complex<float> >& y);
template double RelativeErrorNorm2(Vector< std::complex<double> >& x, Vector< std::complex<double> >& y);

template void SpMV(Sparse<float>& A, Vector<float>& x, Vector<float>& y);
template void SpMV(Sparse<double>& A, Vector<double>& x, Vector<double>& y);
template void SpMV(Sparse< std::complex<float> >& A, Vector< std::complex<float> >& x, Vector< std::complex<float> >& y);
template void SpMV(Sparse< std::complex<double> >& A, Vector< std::complex<double> >& x, Vector< std::complex<double> >& y);
}

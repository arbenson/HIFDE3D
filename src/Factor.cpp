#include "hifde3d.hpp"
#include <iostream>

namespace hifde3d {

template <typename Scalar>
void HIFFactor<Scalar>::Initialize() {
    int NC = N_ + 1;
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::Initialize");
    if (N_ <= 0)
	throw std::logic_error("Number of discretization points must be positive");
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


}

template <typename Scalar>
void HIFFactor<Scalar>::Factor() {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::Factor");
#endif
    int NC = N_ + 1;
    int num_levels = static_cast<int>(round(log2(NC / P_))) + 1;
    schur_level_data_.resize(num_levels);
    skel_level_data_.resize(num_levels);

    for (int level = 0; level < num_levels; ++level) {
        int width = pow2(level) * P_;
        int cells_per_dir = NC / width;

	std::cout << "level " << level << std::endl;
        LevelFactorSchur(cells_per_dir, level);
	std::cout << "Schur done" << std::endl;
        UpdateMatrixAndDOFs(level, false);

        if (level < num_levels - 1) {
            std::cout << "Starting skel..." << std::endl;
            LevelFactorSkel(cells_per_dir, level);
            std::cout << "Skel done" << std::endl;
            UpdateMatrixAndDOFs(level, true);
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
void HIFFactor<Scalar>::UpdateRemainingDOFs(int level, bool is_skel) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::UpdateRemainingDOFs");
#endif
    std::vector<int> eliminated_DOFs;

    // Gather the DOFs
    std::vector< FactorData<Scalar> >& level_data = schur_level_data_[level];
    if (is_skel) {
        level_data = skel_level_data_[level];
    }

    for (size_t i = 0; i < level_data.size(); ++i) {
        std::vector<int>& red_inds = level_data[i].ind_data().redundant_inds();
        std::vector<int>& global_inds = level_data[i].ind_data().global_inds();
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
    InteriorFaceIndexData(cell_location, face, level, skel_data);
    if (skel_data.global_cols().size() == 0) {
        return false;    // No face here.
    }

    Dense<Scalar> submat;
    // std::cout << skel_data.global_rows().size() << " " << skel_data.global_cols().size() << std::endl;
    DenseSubmatrix(sp_matrix_, skel_data.global_rows(), skel_data.global_cols(), submat);
    data.set_face(face);
    std::vector<int>& cols = skel_data.global_cols();
    // TODO: avoid this copy
    std::vector<int>&  global = data.ind_data().global_inds();
    for (size_t i = 0; i < cols.size(); ++i) {
        global.push_back(cols[i]);
    }
    InterpDecomp(submat, data.W_mat(), data.ind_data().skeleton_inds(),
                 data.ind_data().redundant_inds(), epsilon_);
    // std::cout << "interp decomp" << std::endl;
    SchurAfterID(data);
    return true;
}

template <typename Scalar>
void HIFFactor<Scalar>::LevelFactorSchur(int cells_per_dir, int level) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::LevelFactorSchur");
#endif
    std::vector< FactorData<Scalar> >& level_data = schur_level_data_[level];
    int num_DOFs_eliminated = 0;
    for (int i = 0; i < cells_per_dir; ++i) {
        for (int j = 0; j < cells_per_dir; ++j) {
            for (int k = 0; k < cells_per_dir; ++k) {
                FactorData<Scalar> tmp;
                level_data.push_back(tmp);
                FactorData<Scalar>& factor_data = level_data[level_data.size() - 1];
                InteriorCellIndexData(Index3(i, j, k), level, factor_data.ind_data());

		//factor_data.ind_data().Print();

                // Get local data from the global matrix
                std::vector<int>& global_inds = factor_data.ind_data().global_inds();
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
void HIFFactor<Scalar>::LevelFactorSkel(int cells_per_dir, int level) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::LevelFactorSkel");
#endif
    std::vector< FactorData<Scalar> >& level_data = skel_level_data_[level];
    int num_DOFs_eliminated = 0;
    for (int i = 0; i < cells_per_dir; ++i) {
        for (int j = 0; j < cells_per_dir; ++j) {
            for (int k = 0; k < cells_per_dir; ++k) {
                Index3 cell_location(i, j, k);

                // We only do half of the faces since each face is shared by
                // two cells.  To keep indexing consistent, we do the top,
                // front, and left faces.
                // TODO: abstract away these three calls
                {
                    Face face = TOP;
                    level_data.push_back(FactorData<Scalar>());
                    FactorData<Scalar>& top_data = level_data[level_data.size() - 1];
                    top_data.set_face(face);
                    bool ret = Skeletonize(cell_location, face, level, top_data);
                    if (!ret) {
                        level_data.pop_back();
                    } else {
			num_DOFs_eliminated += top_data.NumDOFsEliminated();
		    }
                }

                {
                    Face face = FRONT;
                    level_data.push_back(FactorData<Scalar>());
                    FactorData<Scalar>& front_data = level_data[level_data.size() - 1];
                    front_data.set_face(face);
                    bool ret = Skeletonize(cell_location, face, level, front_data);
                    if (!ret) {
                        level_data.pop_back();
                    } else {
			num_DOFs_eliminated += front_data.NumDOFsEliminated();
		    }
                }

                {
                    Face face = LEFT;
                    level_data.push_back(FactorData<Scalar>());
                    FactorData<Scalar>& left_data = level_data[level_data.size() - 1];
                    left_data.set_face(face);
                    bool ret = Skeletonize(cell_location, face, level, left_data);
                    if (!ret) {
                        level_data.pop_back();
                    } else {
			num_DOFs_eliminated += left_data.NumDOFsEliminated();
		    }
                }
            }
        }
    }

    size_t num_faces_total = 3 * (cells_per_dir - 1) * cells_per_dir * cells_per_dir;
    assert(level_data.size() == num_faces_total);
    // TODO: this counting is wrong
    std::cout << "Level (" << level << ", Skel): "
              << num_DOFs_eliminated << " DOFs eliminated" << std::endl;
}

template <typename Scalar>
void HIFFactor<Scalar>::UpdateMatrixAndDOFs(int level, bool is_skel) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::UpdateMatrixAndDOFs");
#endif
    std::vector< FactorData<Scalar> >& level_data = schur_level_data_[level];
    if (is_skel) {
        level_data = skel_level_data_[level];
    }

    std::map<int, std::pair< Vector<int>, Vector<Scalar> > > vals;
    Vector<int> del_inds;

    // TODO: this process of forming iidx, jidx, and vals could be faster.
    for (size_t n = 0; n < level_data.size(); ++n) {
        FactorData<Scalar>& data = level_data[n];
        IndexData& ind_data = data.ind_data();
        std::vector<int>& skel_inds = ind_data.skeleton_inds();
        std::vector<int>& global_inds = ind_data.global_inds();
        Dense<Scalar>& S = data.Schur_comp();
        assert(S.Height() == S.Width());
        assert(S.LDim() == S.Height());
        assert(S.Height() == static_cast<int>(skel_inds.size()));
        for (size_t i = 0; i < skel_inds.size(); ++i) {
            for (size_t j = 0; j < skel_inds.size(); ++j) {
                vals[global_inds[skel_inds[i]]].first.PushBack
                (global_inds[skel_inds[j]]);
		        vals[global_inds[skel_inds[i]]].second.PushBack(S.Get(i, j));
            }
        }
        // save on storage
        S.Clear();
        std::vector<int>& red_inds = ind_data.redundant_inds();
        for (size_t i = 0; i < red_inds.size(); ++i) {
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
    UpdateRemainingDOFs(level, is_skel);
}

template <typename Scalar>
bool HIFFactor<Scalar>::IsInterior(int level, int a) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::IsInterior");
#endif
    int width = pow2(level) * P_;
    return (a > 0 && a  < N_ && (a % width) != 0);
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
    //std::cout << "min: " << min_inds << std::endl;
    //std::cout << "max: " << max_inds << std::endl;
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
                    if (IsFaceInterior(level, curr_ind)) {
			//std::cout << "skel: " << curr_ind << std::endl;
			global_inds.push_back(Tensor2LinearInd(curr_ind));
			skel_inds.push_back(curr_lin_index);
			++curr_lin_index;
                    } else if (IsCellInterior(level, curr_ind)) {
			//std::cout << "red: " << curr_ind << std::endl;
			global_inds.push_back(Tensor2LinearInd(curr_ind));
			red_inds.push_back(curr_lin_index);
			++curr_lin_index;
                    }
                }
            }
        }
    }
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
void HIFFactor<Scalar>::InteriorFaceIndexData(Index3 cell_location, Face face,
                                              int level, SkelIndexData& data) {
#ifndef RELEASE
    CallStackEntry entry("HIFFactor::InteriorFaceIndexData");
#endif
    // TODO: When we loop over all cells, there is redundant computation.
    //       Here, we just compute what we need for the given cell.
    int width = pow2(level) * P_;
    Index3 min_inds = vec3max(width * cell_location, 1);
    Index3 max_inds = vec3min(width * (cell_location + 1), N_);
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
        // First, all interior faces for current box
        InteriorFaceDOFs(cell_location, TOP, level, global_rows);
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
        InteriorFaceDOFs(cell_location, LEFT, level, global_rows);
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
        InteriorFaceDOFs(cell_location, FRONT, level, global_rows);
        InteriorFaceDOFs(cell_location, BACK, level, global_rows);

        // Now, get all in neighbor cell.
        cell_location(2) -= 1;
        InteriorFaceDOFs(cell_location, TOP, level, global_rows);
        InteriorFaceDOFs(cell_location, BOTTOM, level, global_rows);
        InteriorFaceDOFs(cell_location, LEFT, level, global_rows);
        InteriorFaceDOFs(cell_location, RIGHT, level, global_rows);
        InteriorFaceDOFs(cell_location, FRONT, level, global_rows);
        // BACK has already been counted
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


// Declarations
template class HIFFactor<float>;
template class HIFFactor<double>;
template class HIFFactor< std::complex<float> >;
template class HIFFactor< std::complex<double> >;

}

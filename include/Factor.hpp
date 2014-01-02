#ifndef FACTOR_HPP_
#define FACTOR_HPP_

#include "data.hpp"
#include "dmhm/core/dense.hpp"
#include "dmhm/core/hmat_tools.hpp"
#include "NumTns.hpp"
#include "schur.hpp"
#include "dmhm/core/sparse.hpp"
#include "vec3t.hpp"

#include <vector>

#include <math.h>

template <typename Scalar>
class HIFFactor {
public:
    // TODO: better constructor and destructor
    HIFFactor() {}
    ~HIFFactor() {}

    // Form factorization of the matrix via HIF.
    // The matrix, number of discretiztaion points (N), width at lowest
    // level (P), and target accuracy (epsilon) must already be be provided.
    void Factor();

    // Convert a linear index to a tensor index. The tensor is of size
    // N x N x N, where N is the number of discretization points per direction.
    Index3 Linear2TensorInd(int ind);

    // Convert a tensor index to a linear index. The tensor is of size
    // N x N x N, where N is the number of discretization points per direction.
    int Tensor2LinearInd(Index3 ind);

    void set_N(int N);
    int N();

    void set_P(int P);
    int P();

    void set_epsilon(double epsilon);
    double epsilon();

    dmhm::Sparse<Scalar>& sp_matrix();


private:
    // Remove eliminated degrees of freedom.
    //
    // level (in): level at which to remove DOFs
    // is_skel (in): wheter or not the update is after skeletonization (after
    //               half level)
    void UpdateRemainingDOFs(int level, bool is_skel);

    // Eliminate DOFs interior to cells at a given level.
    //
    // cells_per_dir (in): Number of cells per direction.  There are
    //                     cells_per_dir ^ 3 cells on which to eliminate
    //                     interior DOFs.
    // level (in): level at which to eliminate DOFs
    void LevelFactorSchur(int cells_per_dir, int level);

    // Eliminate DOFs interior to cell faces via skeletonizations.
    //
    // cells_per_dir (in): Number of cells per direction.  There are
    //                     cells_per_dir ^ 3 cells on which to eliminate
    //                     interior DOFs.
    // level (in): level at which to eliminate DOFs
    void LevelFactorSkel(int cells_per_dir, int level);

    // Perform skeletonization on the DOFs corresponding to the
    // interior of a single face.
    //
    // cell_location (in): 3-tuple of cell location
    // Face (in): which face
    // width (in): width of the cell in terms of number of points
    // data (out): fills in all factorization data needed for this DOF set
    //             for solves after factorization completes
    void Skeletonize(Index3 cell_location, Face face, int width,
                         FactorData<Scalar>& data);

    // Eliminate redundant DOFs via Schur complements after the ID has
    // completed.
    // 
    // data (in/out): Needs DOF data (global indices and skeleton/redundant indices)
    //                filled in.  After function completes, all data is filled in.
    void SchurAfterID(FactorData<Scalar>& data);

    // Update the global sparse matrix with the Schur complements from a
    // given level.  Also updates the remaining degrees of freedom.
    // This function should be called after computing all Schur complements
    // from either an integer level (just Schur complements) or half
    // level (skeletonization). 
    //
    // level (in): the level for which computation has completed and for which
    //             data will be used for the update
    // is_skel (in): wheter or not the update is after skeletonization (after
    //               half level)
    void UpdateMatrixAndDOFs(int level, bool is_skel);

    // For a given cell location at a given level, determine the indices of the
    // DOFs interior to the cell.  These DOFs are eliminated by a Schur
    // complement Also, determine the interaction of the interior DOFs.
    // The IndexData is filled with 
    //
    // cell_location (in): 3-tuple of cell location
    // level (in): level of the cell location
    // data (out): fills global indices, DOF set, and DOF set interactions.
    void InteriorCellIndexData(Index3 cell_location, int level, IndexData& data);
    
    // For a given cell location, level, and face, determine the indices of the
    // DOFs interior to the face.  These DOFs are skeletonized.  Also, determine
    // the interaction of the interior face DOFs.
    //
    // cell_location (in): 3-tuple of cell location
    // Face (in): which face
    // level (in): level of the cell location
    // data (out): fills global row and global column indices for skeletonization
    void InteriorFaceIndexData(Index3 cell_location, Face face, int level,
                               SkelIndexData& data);

    // Determine whether an index is on the interior of a cell.
    //
    // level (in): level of the partition
    // ind (in): index
    bool IsCellInterior(int level, Index3 ind);

    // Determine whether an index is on the interior at a face.
    //
    // level (in): level of the partition
    // ind (in): index
    bool IsFaceInterior(int level, Index3 ind);

    // Determine whether an index is on the interior at a given level.
    //
    // level (in): level of the partition
    // ind (in): index
    bool IsInterior(int level, int a);

    // DATA
    dmhm::Sparse<Scalar> sp_matrix_;
    int N_;
    int P_;
    double epsilon_;
    IntNumTns remaining_DOFs_;
    std::vector< std::vector< FactorData<Scalar> > > schur_level_data_;
    std::vector< std::vector< FactorData<Scalar> > > skel_level_data_;
};

template <typename Scalar>
void HIFFactor<Scalar>::Factor() {
    int NC = N_ + 1;
    int num_levels = reinterpret_cast<int>(round(log2(NC / P_))) + 1;

    for (int level = 0; level < num_levels; ++level) {
        int width = pow2(level) * P_;
        int cells_per_dir = NC / width;

        LevelFactorSchur(cells_per_dir, width, level);
        UpdateMatrixAndDOFs(level, false);
        
        if (level < num_levels - 1) {
            LevelFactorSchur(cells_per_dir, width, level);
            UpdateMatrixAndDOFs(level, true);
        }
    }
}

template <typename Scalar>
Index3 HIFFactor<Scalar>::Linear2TensorInd(int ind) {
    int i = ind % N_;
    int j = ((ind - i) % (N_ * N_)) / N_;
    int k = (ind - i - N_ * j) / (N_ * N_);
    assert(0 <= i && i < N_ && 0 <= j && j < N_ && 0 <= k && k < N_);
    return Index3(i, j, k);
}

template <typename Scalar>
int HIFFactor<Scalar>::Tensor2LinearInd(Index3 ind) {
    int i = ind(0);
    int j = ind(1);
    int k = ind(2);
    assert(0 <= i && i < N_ && 0 <= j && j < N_ && 0 <= k && k < N_);
    return i + N * j + (N * N) * k;
}

template <typename Scalar>
void HIFFactor<Scalar>::UpdateRemainingDOFs(int level, bool is_skel) {
    std::vector<int> eliminated_DOFs;

    // Gather the DOFs
    std::vector< FactorData<Scalar> >& level_data;
    if (is_skel) {
        level_data = skel_level_data_[level];
    } else {
        level_data = schur_level_data_[level];
    }
    for (int i = 0; i < level_data.Size(); ++i) {
	std::vector<int>& DOF_set = level_data[i].ind_data().DOF_set();
	std::vector<int>& global_inds = level_data[i].ind_data().global_inds();
        for (int j = 0; j < DOF_set.size(); ++j) {
            eliminated_DOFs.push_back(global_inds[DOF_set[j]]);
        }
    }

    // Eliminate DOFs
    for (int i = 0; i < eliminated_DOFs.size(); ++i) {
        Index3 ind = Linear2TensorInd(eliminated_DOFs[i]);
        remaining_DOFs_(ind(0), ind(1), ind(2)) = 0;
    }
}

template <typename Scalar>
void HIFFactor<Scalar>::SchurAfterID(FactorData<Scalar>& data) {
    std::vector<int>& global_inds = data.ind_data().global_inds();
    dmhm::Dense<Scalar> submat;
    DenseSubmatrix(sp_matrix_, global_inds, global_inds, submat);

    // Start with identity
    int size = global_inds.size();
    dmhm::Dense<Scalar> Rot(size, size, dmhm::GENERAL);
    for (int i = 0; i < size; ++i) {
        Rot(i, i) = Scalar(1);
    }

    // Fill in with W
    std::vector<int>& skeleton_inds = data.ind_data().skeleton_set();
    std::vector<int>& redundant_inds = data.ind_data().redundant_set();
    for (int i = 0; i < skeleton_inds.size(); ++i) {
        for (int j = 0; j < redundant_inds.size(); ++j) {
            Rot(skeleton_inds[i], redundant_inds[j]) = -data.W_mat()(i, j);
        }
    }

    dmhm::Dense<Scalar> tmp(submat.Height(), Rot.Width(), dmhm::GENERAL);
    dmhm::hmat_tools::Multiply(Scalar(1), submat, Rot, tmp);
    
    dmhm::Dense<Scalar> result(Rot.Height(), tmp.Width(), dmhm::GENERAL);
    dmhm::hmat_tools::AdjointMultiply(Scalar(1), Rot, tmp, result);
    Schur(result, data);
}

template <typename Scalar>
void HIFFactor<Scalar>::Skeletonize(Index3 cell_location, Face face,
                                        int width, FactorData<Scalar>& data) {
    SkelIndexData skel_data;
    InteriorFaceIndexData(cell_location, face, width, skel_data);
    
    dmhm::Dense<Scalar> submat;
    DenseSubmatrix(sp_matrix_, skel_data.global_rows(), skel_data.global_cols(), submat);
    data.set_face(face);
    std::vector<int>& cols = skel_data.global_cols();
    // TODO: avoid this copy
    std::vector<int>&  global = data.global_inds();
    for (int i = 0; i < cols.size(); ++i) {
        global.push_back(cols[i]);
    }
    InterpDecomp(submat, data.W_mat(), data.ind_data().skeleton_set(),
                 data.ind_data().redundant_set(), epsilon_);
    SchurAfterID(data);
}

template <typename Scalar>
void HIFFactor<Scalar>::LevelFactorSchur(int cells_per_dir, int level) {
    std::vector< FactorData<Scalar> >& level_data = schur_level_data_[level];
    for (int i = 0; i < cells_per_dir; ++i) {
        for (int j = 0; i < cells_per_dir; ++j) {
            for (int k = 0; i < cells_per_dir; ++k) {
		FactorData<Scalar> tmp;
                level_data.push_back(tmp);
                FactorData<Scalar>& factor_data = level_data[level_data.size() - 1];
                InteriorCellIndexData(Index3(i, j, k), level, factor_data.ind_data());
                
                // Get local data from the global matrix
                std::vector<int>& global_inds = factor_data.ind_data().global_inds();
                dmhm::Dense<Scalar> submat;
                DenseSubmatrix(sp_matrix_, global_inds, global_inds, submat);
                Schur(submat, factor_data);
            }
        }
    }
}

template <typename Scalar>
void HIFFactor<Scalar>::LevelFactorSkel(int cells_per_dir, int level) {
    std::vector< FactorData<Scalar> >& level_data = skel_level_data_[level];
    std::vector<Index3> eliminated_DOFs;
    for (int i = 0; i < cells_per_dir; ++i) {
        for (int j = 0; i < cells_per_dir; ++j) {
            for (int k = 0; i < cells_per_dir; ++k) {
                Index3 cell_location(i, j, k);

                // We only do half of the faces since each face is shared by
                // two cells.  By arbitrary choice, we do the top, front, and
                // right faces.
                Face face = TOP;
                level_data.push_back(FactorData<Scalar>());
                FactorData<Scalar>& top_data = level_data[level_data.size() - 1];
                top_data.set_face(face);
                Skeletonize(face, cell_location, top_data);

                face = FRONT;
                level_data.push_back(FactorData<Scalar>());
                FactorData<Scalar>& front_data = level_data[level_data.size() - 1];
                front_data.set_face(face);
                Skeletonize(face, cell_location, front_data);

                face = RIGHT;
                level_data.push_back(FactorData<Scalar>());
                FactorData<Scalar>& right_data = level_data[level_data.size() - 1];
                right_data.set_face(face);
                Skeletonize(face, cell_location, right_data);
            }
        }
    }
}

template <typename Scalar>
void HIFFactor<Scalar>::UpdateMatrixAndDOFs(int level, bool is_skel) {
    // TODO: implement this function
    return;
}

template <typename Scalar>
bool HIFFactor<Scalar>::IsInterior(int level, int a) {
    int width = pow2(level) * P_;
    return (a == 0 || a == N_ - 1 || (a % width) != 0);
}


template <typename Scalar>
bool HIFFactor<Scalar>::IsFaceInterior(int level, Index3 ind) {
    int a_int = IsInterior(level, ind(0));
    int b_int = IsInterior(level, ind(1));
    int c_int = IsInterior(level, ind(2));
    return ((a_int && b_int && !c_int) ||
            (a_int && !b_int && c_int) ||
	    (!a_int && b_int && c_int));
}

template <typename Scalar>
bool HIFFactor<Scalar>::IsCellInterior(int level, Index3 ind) {
    return IsInterior(level, ind(0)) &&
           IsInterior(level, ind(1)) &&
	   IsInterior(level, ind(2));
}

template <typename Scalar>
void HIFFactor<Scalar>::InteriorCellIndexData(Index3 cell_location, int level,
                                              IndexData& data) {
    int width = pow2(level) * P_;
    Index3 min_inds = width * cell_location;
    Index3 max_inds = vec3max(width * (cell_location + 1), N_ - 1);
    assert(min_inds <= max_inds);
    assert(min_inds >= Index3(0, 0, 0));

    std::vector<int>& DOF_set = data.DOF_set();
    std::vector<int>& DOF_set_interaction = data.DOF_set_interaction();
    for (int i = min_inds(0); i < max_inds(0); ++i) {
        for (int j = min_inds(1); i < max_inds(1); ++j) {
            for (int k = min_inds(2); i < max_inds(2); ++k) {
		if (remaining_DOFs_(i, j, k)) {
		    Index3 curr_ind(i, j, k);
		    if (IsFaceInterior(level, curr_ind)) {
			DOF_set_interaction.push_back(Tensor2LinearInd(curr_ind));
		    } else if (IsCellInterior(level, curr_ind)) {
			DOF_set.push_back(Tensor2LinearInd(curr_ind));
		    }
		}
            }
        }
    }
}

template <typename Scalar>
void HIFFactor<Scalar>::InteriorFaceIndexData(Index3 cell_location, Face face,
                                              int level, SkelIndexData& data) {
    // TODO: implement this function
    int width = pow2(level) * P_;
    /*
    switch (face) {
    case TOP:
	// first index is fixed
	Index3 inds = cell_location * width;
	int i = inds(0);
	int j_min = inds(1);
	int j_max = max(inds(1) + width, N - 1);
	int k_min = inds(2);
	int k_max = max(inds(2) + width, N - 1);
	// face indices: (i, j_min:j_max, k_min:k_max)
    }
    */
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
dmhm::Sparse<Scalar>& HIFFactor<Scalar>::sp_matrix() { return sp_matrix_; }

#endif  // ifndef FACTOR_HPP_

#ifndef FACTOR_HPP_
#define FACTOR_HPP_

#include "data.hpp"
#include "dense.hpp"
#include "sparse.hpp"
#include "vec3t.hpp"
#include "vector.hpp"

#include "math.h"


template <typename Scalar>
class HIFFactor {
public:
    // TODO: constructor and destructor
    int Factor();

private:
    int LevelFactorSchur();
    int LevelFactorSkel();
    int FormSchurAfterID();

    // Take the (dense) Schur complement and update the matrix.
    // Also update the matrix based on the eliminated degrees of freedom.
    int UpdateMatrix(Dense<Scalar> schur_comp, Vector<int>& rows,
		     Vector<int>& cols, Vector<Index3>& eliminated_DOFs);

    // For a given cell location at a given level, determine the indices of the
    // DOFs interior to the cell.  These DOFs are eliminated by a Schur
    // complement Also, determine the interaction of the interior DOFs.
    //
    // cell_location (in): 3-tuple of cell location
    // W (in): width of the cell
    // return value: 0 on failure, 1 on success
    int InteriorCellIndexData(Index3 cell_location, int W, IndexData& data);
    
    // For a given cell location, level, and face, determine the indices of the
    // DOFs interior to the face.  These DOFs are skeletonized.  Also, determine
    // the interaction of the interior face DOFs.
    //
    // cell_location (in): 3-tuple of cell location
    // Face (in): which face
    // W (in): width of the cell
    // data (out): indexing data
    // return value: 0 on failure, 1 on success
    int InteriorFaceIndexData(Index3 cell_location, Face face, int W, IndexData& data);

    Sparse<Scalar> sp_matrix_;
    int N_;
    int P_;
    double epsilon_;
    IntNumTns remaining_DOFs_;
    Vector< Vector< FactorData<Scalar> > > schur_level_data_;
    Vector< Vector< FactorData<Scalar> > > skel_level_data_;
}

// TODO: re-organize the push_backs to avoid copies

template <typename Scalar>
int HIFFactor<Scalar>::Factor() {
    // ROUGH OUTLINE OF THIS FUNCTION
    int NC = N_ + 1;
    int num_levels = reinterpret_cast<int>(round(log2(NC / P_))) + 1;

    for (int level = 0; level < num_levels; ++level) {
	int W = pow2(level - 1) * P_;
        int points_per_cell = NC / W;

	auto curr_data = schur_level_data_[level];
	LevelFactorSchur(W, curr_data);
	UpdateRemainingDOFsSchur(level);
	UpdateMatrix(sp_matrix_, curr_data);
	
	if (level < num_levels - 1) {
	    auto curr_data = skel_level_data_[level];
	    LevelFactorSkel(W, curr_data);
	    UpdateRemainingDOFsSkel(level);
	    UpdateMatrix(sp_matrix_, curr_data);
	}
    }
}

template <typename Scalar>
Index3 HIFFactor<Scalar>::Linear2TensorInd(int lin_ind) {
    // Convert linear index to Index3 of N x N x N
    // TODO: implement this function
}

template <typename Scalar>
int HIFFactor<Scalar>::UpdateRemainingDOFs(Vector<int>& eliminated_DOFs) {
    for (int i = 0; i < eliminated_DOFs.size(); ++i) {
	Index3 ind = Linear2TensorInd(eliminated_DOFs[i]);
	remaining_DOFs_(ind(0), ind(1), ind(2)) = 0;
    }
}

template <typename Scalar>
int HIFFactor<Scalar>::UpdateRemainingDOFsSchur(int level) {
    Vector<int> eliminated_DOFs;
    auto level_data = schur_level_data_[level];
    for (int i = 0; i < level_data.Size(); ++i) {
	auto DOF_set = level_data[i].ind_data().DOF_set();
	auto global_inds = level_data[i].ind_data().global_inds();
	for (int j = 0; j < DOF_set.Size(); ++j) {
	    eliminated_DOFs.PushBack(global_inds[DOF_set[j]]);
	}
    }
    UpdateRemainingDOFs(eliminated_DOFs);
}

template <typename Scalar>
int HIFFactor<Scalar>::UpdateRemainingDOFsSkel(int level) {
    Vector<Int> eliminated_DOFs;
    auto level_data = skel_level_data_[level];
    for (int i = 0; i < level_data.Size(); ++i) {
	auto DOF_set = level_data[i].ind_data().DOF_set();
	auto global_inds = level_data[i].ind_data().global_inds();
	for (int j = 0; j < DOF_set.Size(); ++j) {
	    eliminated_DOFs.PushBack(global_inds[DOF_set[j]]);
	}
    }
    UpdateRemainingDOFs(eliminated_DOFs);
}

template <typename Scalar>
int HIFFactor<Scalar>::LevelFactorSchur(int cells_per_dir, int W,
					Vector< FactorData<Scalar> >& level_data) {
    for (int i = 0; i < cells_per_dir; ++i) {
	for (int j = 0; i < cells_per_dir; ++j) {
	    for (int k = 0; i < cells_per_dir; ++k) {
		FactorData<Scalar> factor_data;
		InteriorCellIndexData(Index3(i, j, k), W, N, factor_data.ind_data());
		
		// Get local data from the global matrix
		auto global_inds = factor_data.ind_data().global_inds();
		Dense<Scalar> submat;
		DenseSubmatrix(sp_matrix_, global_inds, global_inds, submat);

		// Eliminate DOFs via Schur complement
		FactorData<Scalar> factor_data;
		
		Schur(submat, factor_data);
		level_data.push_back(factor_data);
	    }
	}
    }
}

template <typename Scalar>
int HIFFactor<Scalar>::FormSchurAfterID(FactorData<Scalar>& data) {
    auto global_inds = data.ind_data().global_inds();
    Dense<Scalar> submat;
    DenseSubmatrix(sp_matrix_, global_inds, global_inds, submat);

    int size = global_inds.Size();
    Dense<Scalar> Rot = Dense(size, size, GENERAL);
    for (int i = 0; i < size; ++i) {
	Rot(i, i) = One<Scalar>();
    }
    auto skeleton_inds = data.ind_data().skeleton_set();
    auto redundant_inds = data.ind_data().redundant_set();
    for (int i = 0; i < skeleton_inds.size(); ++i) {
	for (int j = 0; j < redundant_inds.size(); ++j) {
	    Rot(skeleton_inds[i], redundant_inds[j]) = data.W_mat()(i, j);
	}
    }

    // TODO: check to make sure assignment like this works
    submat.Multiply(One<Scalar>, Rot, submat);
    Rot.HermitianTransposeMultiply(One<Scalar>, submat, submat);
    Schur(submat, redundant_inds, skeleton_inds, data);
}

template <typename Scalar>
int HIFFactor<Scalar>::Skeletonization(Face face, Index3 cell_location, FactorData<Scalar>& data) {
    SkelIndexData skel_data;
    InteriorCellIndexData(cell_location, face, W, skel_data);
    
    Dense<Scalar> submat;
    DenseSubmatrix(sp_matrix_, skel_data.global_rows(), skel_data.global_cols(), submat);
    FactorData<Scalar> factor_data;
    data.set_face(face);
    InterpDecomp(submat, factor_data.W_mat(), factor_data.ind_data().skeleton_set(),
		 factor_data.ind_data().redundant_set(), epsilon_);
    FormSchurAfterID(skeleton_cols, redundant_cols, data.global_cols(), data);
}

template <typename Scalar>
int HIFFactor<Scalar>::LevelFactorSkel(int cells_per_dir, int W,
				       Vector<SkelData>& level_data) {
    Vector<Index3> eliminated_DOFs;
    for (int i = 0; i < cells_per_dir; ++i) {
	for (int j = 0; i < cells_per_dir; ++j) {
	    for (int k = 0; i < cells_per_dir; ++k) {
		Index3 cell_location(i, j, k);

		// Note: we only do half of the faces
		Face face = Face::TOP;
		FactorData<Scalar> data_top;
		data.set_face(face);
		Skeletonization(face, cell_location, data_top);
		level_data.push_back(data_top);

		Face face = Face::FRONT;
		FactorData<Scalar> data_front;
		data.set_face(face);
		Skeletonization(face, cell_location, data_front);
		level_data.push_back(data_front);

		Face face = Face::RIGHT;
		FactorData<Scalar> data_right;
		data.set_face(face);
		Skeletonization(face, cell_location, data_right);
		level_data.push_back(data_right);
	    }
	}
    }
    return 0;
}

template <typename Scalar>
int HIFFactor<Scalar>::UpdateMatrix() {
    // TODO: implement this function
    return 0;
}

#endif  // ifndef FACTOR_HPP_

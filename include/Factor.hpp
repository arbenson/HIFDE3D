#ifndef FACTOR_HPP_
#define FACTOR_HPP_

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

    Sparse<Scalar> sp_matrix_;
    int N_;
    int P_;
    double epsilon_;
    IntNumTns remaining_DOFs_;
}

// TODO: re-organize the push_backs to avoid copies

template <typename Scalar>
int HIFFactor<Scalar>::Factor() {
    // ROUGH OUTLINE OF THIS FUNCTION
    int NC = N_ + 1;
    int num_levels = reinterpret_cast<int>(round(log2(NC / P_))) + 1;

    Vector< Vector< FactorData<Scalar> > > SchurLevelData(num_levels);
    Vector< Vector< FactorData<Scalar> > > SkelLevelData(num_levels);

    for (int level = 0; level < NL; ++level) {
	int W = pow2(level - 1) * P_;
        int points_per_cell = NC / W;

	auto curr_data = SchurLevelData[level];
	LevelFactorSchur(W, curr_data);
	UpdateMatrix(sp_matrix_, curr_data);
	
	if (level < NL - 1) {
	    auto curr_data = SkelLevelData[i];
	    LevelFactorSkel(W, curr_data);
	    UpdateMatrix(sp_matrix_, curr_data);
	}
    }
}

template <typename Scalar>
int HIFFactor<Scalar>::LevelFactorSchur(int cells_per_dir, int W,
				   Vector< FactorData<Scalar> >& level_data) {
    for (int i = 0; i < cells_per_dir; ++i) {
	for (int j = 0; i < cells_per_dir; ++j) {
	    for (int k = 0; i < cells_per_dir; ++k) {
		IndexData ind_data;
		InteriorCellIndexData(Index3(i, j, k), W, N, remaining_DOFs, ind_data);
		
		Dense<Scalar> submat;
		DenseSubmatrix(sp_matrix_, data.global_inds(), data.global_inds(), submat);

		FactorData<Scalar> factor_data;
		Schur(submat, data.DOF_set(), data.DOF_set_interaction(), factor_data);
		level_data.push_back(factor_data);

		// Update eliminated DOFs
	    }
	}
    }
}

template <typename Scalar>
int HIFFactor<Scalar>::FormSchurAfterID(Vector<int>& skeleton_inds,
                                   Vector<int>& redundant_inds,
                                   Vector<int>& global_inds,
				   FactorData<Scalar>& data) {
    Dense<Scalar> submat;
    DenseSubmatrix(sp_matrix_, global_inds, global_inds, submat);

    /* 
       TODO: Implement this
       Rot = eye(numel(global_inds));
       Rot(lclsk,lclrd) = -lclW;
       submat = Rot'*submat*Rot;
    */

    Schur(submat, redundant_inds, skeleton_inds, data);
}

template <typename Scalar>
int HIFFactor<Scalar>::Skeletonization(Face face, Index3 cell_location, FactorData<Scalar>& data) {
    SkelIndexData skel_data;
    InteriorCellIndexData(cell_location, face, W, N_, remaining_DOFs_, skel_data);
    
    Dense<Scalar> submat;
    DenseSubmatrix(sp_matrix_, skel_data.global_rows(), skel_data.global_cols(), submat);
    Dense<Scalar> lclW;
    Vector<int> skeleleton_cols;
    Vector<int> redundant_cols;
    InterpDecomp(submat, lclW, skeleton_cols, redundant_cols, epsilon_);
    
    FactorData<Scalar> data;
    data.set_W_mat(lclW);
    data.set_face(face);
    FormSchurAfterID(skeleton_cols, redundant_cols, data.global_cols(), data);
}

template <typename Scalar>
int HIFFactor<Scalar>::LevelFactorSkel(int cells_per_dir, int W, Vector<SkelData>& level_data) {
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

		// Update eliminated DOFs
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

#endif  // FACTOR_HPP_

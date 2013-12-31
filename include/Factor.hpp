#ifndef FACTOR_HPP_
#define FACTOR_HPP_

#include "math.h"

template <class T>
class HIFFactor {
public:
    // TODO: constructor and destructor

    int Factor();

private:
    int LevelFactorSchur();
    int LevelFactorSkel();
    int UpdateMatrix();

    Sparse<T> sp_matrix_;
    int N_;
    int P_;
    double epsilon_;
    IntNumTns remaining_DOFs_;
}

template <class T>
int HIFFactor<T>::Factor() {
    // ROUGH OUTLINE OF THIS FUNCTION
    int NC = N_ + 1;
    int num_levels = reinterpret_cast<int>(round(log2(NC / P_))) + 1;

    std::vector< std::vector<SchurData> > SchurLevelData(num_levels);
    std::vector< std::vector<SkelData> > SkelLevelData(num_levels);

    for (int i = 0; i < NL; ++i) {
	int W = pow2(ell-1) * P_;
        int points_per_cell = NC / W;

	auto curr_data = SchurLevelData[i];
	LevelFactorSchur(W);
	UpdateMatrix(sp_matrix_, curr_data);

	auto curr_data = SkelLevelData[i];
	LevelFactorSkel(W);
	UpdateMatrix(sp_matrix_, curr_data);
    }
}

template <class T>
int HIFFactor<T>::LevelFactorSchur(int cells_per_dir, int W) {
    // TODO: implement this function
    // ROUGH OUTLINE OF THIS FUNCTION
    for (int i = 0; i < cells_per_dir; ++i) {
	for (int j = 0; i < cells_per_dir; ++i) {
	    for (int k = 0; i < cells_per_dir; ++i) {
		IndexData ind_data;
		InteriorCellIndexData(Index3(i, j, k), W, N, remaining_DOFs, ind_data);
		
		NumMat submat;
		DenseSubmatrix(sp_matrix_, data.global_inds(), data.global_inds(), submat);

		SchurData schur_data;
		Schur(submat, data.DOF_set(), data.DOF_set_interaction(), schur_data);
		level_data.push_back(schur_data);
	    }
	}
    }
}

template <class T>
int HIFFactor<T>::LevelFactorSkel(int cells_per_dir, int W) {
    // TODO: implement this function
    // ROUGH OUTLINE OF THIS FUNCTION
}

template <class T>
int HIFFactor<T>::UpdateMatrix() {

}


#endif  // FACTOR_HPP_

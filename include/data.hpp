#ifndef DATA_HPP_
#define DATA_HPP_

#include "hifde3d.hpp"

// Faces of a cube
enum Face {TOP=0, BOTTOM, RIGHT, LEFT, FRONT, BACK};

namespace hifde3d {

class IndexData {
public:
    IndexData() {}
    ~IndexData() {
	global_inds_.clear();
	redundant_inds_.clear();
	skeleton_inds_.clear();
    }

    // Indices into the global matrix of size N^3 x N^3
    std::vector<int>& global_inds() { return global_inds_; }

    // If global_inds_ is of size n, then the redundant and skeleton
    // indices are disjoint index subsets of {0, ..., n-1} that correspond
    // to the degrees of freedom being eliminated and their interactions.
    std::vector<int>& redundant_inds() { return redundant_inds_; }
    std::vector<int>& skeleton_inds() { return skeleton_inds_; }

    void Print() {
	std::cout << "Global indices: " << std::endl;
	for (size_t i = 0; i < global_inds_.size(); ++i) {
	    std::cout << global_inds_[i] << std::endl;
	}
	std::cout << "Redundant indices: " << std::endl;
	for (size_t i = 0; i < redundant_inds_.size(); ++i) {
	    std::cout << redundant_inds_[i] << std::endl;
	}
	std::cout << "Skeleton indices: " << std::endl;
	for (size_t i = 0; i < skeleton_inds_.size(); ++i) {
	    std::cout << skeleton_inds_[i] << std::endl;
	}
    }

    void PrintGlobal() {
	std::cout << "Redundant (global): " << std::endl;
	for (size_t i = 0; i < redundant_inds_.size(); ++i) {
	    std::cout << global_inds_[redundant_inds_[i]] << std::endl;
	}
	std::cout << "Skeleton (global): " << std::endl;
	for (size_t i = 0; i < redundant_inds_.size(); ++i) {
	    std::cout << global_inds_[skeleton_inds_[i]] << std::endl;
	}
    }

private:
    std::vector<int> global_inds_;         // indices into N^3 x N^3 system
    std::vector<int> redundant_inds_;      // indices of global_inds_ corresponding
                                           // to what is being eliminated
    std::vector<int> skeleton_inds_;       // indices of global_inds_ corresponding
                                           // to non-zero entries of the matrix below
                                           // global_inds_(redundant_inds_).
};

class SkelIndexData {
public:
    SkelIndexData() {}
    ~SkelIndexData() {
	global_rows_.clear();
	global_cols_.clear();
    }

    // Indices into the global matrix of size N^3 x N^3
    std::vector<int>& global_rows() { return global_rows_; }
    std::vector<int>& global_cols() { return global_cols_; }

private:
    std::vector<int> global_rows_;
    std::vector<int> global_cols_;
};

// Structure for storing the Schur complement data needed for application
// to a vector.
template <typename Scalar>
class FactorData {
public:
    Dense<Scalar>& A_22() { return A_22_; }
    Dense<Scalar>& A_22_inv() { return A_22_inv_; }
    Dense<Scalar>& X_mat() { return X_mat_; }
    Dense<Scalar>& Schur_comp() { return Schur_comp_; }
    Dense<Scalar>& W_mat() { return W_mat_; }

    int NumDOFsEliminated() { return ind_data_.redundant_inds().size(); }

    IndexData& ind_data() { return ind_data_; }
    void set_face(Face face) { face_ = face; }
    Face face() { return face_; }

    bool HasZeroDim() { return ind_data_.redundant_inds().size() == 0
	    || ind_data_.skeleton_inds().size() == 0; }

private:
    IndexData ind_data_;
    Dense<Scalar> A_22_;        // matrix restricted to interactions
    Dense<Scalar> A_22_inv_;    // explicit inverse of A_22
    Dense<Scalar> X_mat_;       // A_22_inv * A_21
    Dense<Scalar> Schur_comp_;  // -A_12 * X_mat
    Dense<Scalar> W_mat_;       // Interpolative factor (only for Skel)
    Face face_;                 // To which face this data corresponds
                                // (only for Skel)
};

}
#endif  // ifndef DATA_HPP_

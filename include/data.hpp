#ifndef DATA_HPP_
#define DATA_HPP_

#include "dmhm/core/dense.hpp"

// Faces of a cube
enum Face {TOP=0, BOTTOM, RIGHT, LEFT, FRONT, BACK};

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

private:
    std::vector<int> global_inds_;         // indices into N^3 x N^3 system
    std::vector<int> redundant_inds_;      // indices of global_inds_ corresponding
                                           // to what is being eliminated
    std::vector<int> skeleton_inds_;        // indices of global_inds_ corresponding
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
    dmhm::Dense<Scalar>& A_22() { return A_22_; }
    dmhm::Dense<Scalar>& A_22_inv() { return A_22_inv_; }
    dmhm::Dense<Scalar>& X_mat() { return X_mat_; }
    dmhm::Dense<Scalar>& Schur_comp() { return Schur_comp_; }
    dmhm::Dense<Scalar>& W_mat() { return W_mat_; }

    int NumDOFsEliminated() { return ind_data_.skeleton_inds().size(); }

    IndexData& ind_data() { return ind_data_; }
    void set_face(Face face) { face_ = face; }
    Face face() { return face_; }

private:
    IndexData ind_data_;
    dmhm::Dense<Scalar> A_22_;        // matrix restricted to interactions
    dmhm::Dense<Scalar> A_22_inv_;    // explicit inverse of A_22
    dmhm::Dense<Scalar> X_mat_;       // A_22_inv * A_21
    dmhm::Dense<Scalar> Schur_comp_;  // -A_12 * X_mat
    dmhm::Dense<Scalar> W_mat_;       // Interpolative factor (only for Skel)
    Face face_;                       // to which face this data corresponds
                                      // (only for Skel)
};

#endif  // ifndef DATA_HPP_

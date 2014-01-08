#ifndef DATA_HPP_
#define DATA_HPP_

#include "hifde3d/core/dense.hpp"

// Faces of a cube
enum Face {TOP=0, BOTTOM, RIGHT, LEFT, FRONT, BACK};

namespace hifde3d {

class IndexData {
public:
    IndexData() {}
    ~IndexData() {
	global_inds_.clear();
	DOF_set_.clear();
	DOF_set_interaction_.clear();
    }

    // Indices into the global matrix of size N^3 x N^3
    std::vector<int>& global_inds() { return global_inds_; }

    // If global_inds_ is of size n, then the DOF_set_* vectors
    // are disjoint index subsets of {0, ..., n-1} that correspond
    // to the degrees of freedom and the interactions.
    std::vector<int>& DOF_set() { return DOF_set_; }
    std::vector<int>& DOF_set_interaction() { return DOF_set_interaction_; }

    // Aliases for above to make the code readable.
    std::vector<int>& redundant_set() { return DOF_set_; }
    std::vector<int>& skeleton_set() { return DOF_set_interaction_; }

private:
    std::vector<int> global_inds_;         // indices into N^3 x N^3 system
    std::vector<int> DOF_set_;             // indices of global_inds_ corresponding
                                           // to what is being eliminated
    std::vector<int> DOF_set_interaction_; // indices of global_inds_ corresponding
                                           // to non-zero entries of the matrix below
                                           // global_inds_(DOF_set_).
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

    int NumDOFsEliminated() { return ind_data_.DOF_set().size(); }

    IndexData& ind_data() { return ind_data_; }
    void set_face(Face face) { face_ = face; }
    Face face() { return face_; }

private:
    IndexData ind_data_;
    Dense<Scalar> A_22_;        // matrix restricted to interactions
    Dense<Scalar> A_22_inv_;    // explicit inverse of A_22
    Dense<Scalar> X_mat_;       // A_22_inv * A_21
    Dense<Scalar> Schur_comp_;  // -A_12 * X_mat
    Dense<Scalar> W_mat_;       // Interpolative factor (only for Skel)
    Face face_;                 // to which face this data corresponds (only for Skel)
};

}
#endif  // ifndef DATA_HPP_

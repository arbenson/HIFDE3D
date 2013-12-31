#ifndef _DATA_HPP_
#define _DATA_HPP_

#include "dense.hpp"
#include "vector.hpp"

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

class IndexData {
public:
    ~IndexData() {
	global_inds_.clear();
	DOF_set_.clear();
	DOF_set_interaction_.clear();
    }

    // Indices into the global matrix of size N^3 x N^3
    Vector<int>& global_inds() { return global_inds_; }

    // If global_inds_ is of size n, then the DOF_set_* vectors
    // are disjoint index subsets of {0, ..., n-1} that correspond
    // to the degrees of freedom and the interactions.
    Vector<int>& DOF_set() { return DOF_set_; }
    Vector<int>& DOF_set_interaction() { return DOF_set_interaction_; }

    // Aliases for above to make the code readable.
    Vector<int>& redundant_set() { return DOF_set_; }
    Vector<int>& skeleton_set() { return DOF_set_interaction_; }

private:
    Vector<int> global_inds_;
    Vector<int> DOF_set_;    
    Vector<int> DOF_set_interaction_;
};

class SkelIndexData {
public:
    ~IndexData() {
	global_rows_.clear();
	global_cols_.clear();
    }

    // Indices into the global matrix of size N^3 x N^3
    Vector<int>& global_rows() { return global_rows_; }
    Vector<int>& global_rows() { return global_cols_; }

private:
    Vector<int> global_rows_;
    Vector<int> global_cols_;
};

enum class Face {TOP, BOTTOM, RIGHT, LEFT, FRONT, BACK};

#endif  // ifndef DATA_HPP_

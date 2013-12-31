#ifndef _DATA_HPP_
#define _DATA_HPP_

// Structure for storing the Schur complement data needed for application
// to a vector.
template <class T>
class SchurData {
public:
    // After A_22 has been set, fill in inverse, X_mat, and Schur update
    void FillData(NumMat<T> A_21);

    void set_A_22(NumMat<T> A_22);

private:
    IndexData data_;
    NumMat<T> A_22_;            // matrix restricted to interactions
    NumMat<T> A_22_inv_;        // explicit inversion of A_22
    NumMat<T> X_mat_;           // A_22_inv * A_21
    NumMat<T> Schur_update_;    // -A_12 * X_mat
};

// Structure for storing the skeletonization data needed for application
// to a vector.
template <class T>
class SkeletonizationData {
};

#endif  // _DATA_HPP_

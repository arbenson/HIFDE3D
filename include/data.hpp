#ifndef _DATA_HPP_
#define _DATA_HPP_

// Structure for storing the Schur complement data needed for application
// to a vector.
template <typename Scalar>
class FactorData {
public:
    void set_A_22(Dense<Scalar>& A_22);
    void set_W_mat(Dense<Scalar>& W_mat);
    void set_face(Face face) { face_ = face; }

private:
    IndexData data_;
    Dense<Scalar> A_22_;            // matrix restricted to interactions
    Dense<Scalar> A_22_inv_;        // explicit inversion of A_22
    Dense<Scalar> X_mat_;           // A_22_inv * A_21
    Dense<Scalar> Schur_update_;    // -A_12 * X_mat
    Dense<Scalar> W_mat_;           // Interpolative factor (only for Skel)
    Face face_;
};

#endif  // _DATA_HPP_

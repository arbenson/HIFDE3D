#include "Factor.hpp"
#include "data.hpp"
#include "dmhm/core/vector.hpp"

// Obtain u_skel = u(skel), where skel is the set of skeleton DOFs.
// Here, 'skeleton DOFs' refers to DOFs _not_ being eliminated.
// The skeleton points could be the interior of faces _or_ the skeleton
// points from skeletonization.
//
// u (in): vector of size N^3
// data (in): factor data that contains the skeleton indices
// skel_vec (out): u at the skeleton points
template <typename Scalar>
void GetSkeletonVector(dmhm::Vector<Scalar>& u, FactorData<Scalar>& data,
                       dmhm::Vector<Scalar>& skel_vec) {
    std::vector<int> global_inds = data.global_inds();
    std::vector<int> skel_inds = data.skeleton_set();
    skel_vec.Resize(skel_inds.size());
    for (size_t i = 0; i < skel_inds.size(); ++i) {
        skel_vec.Set(i, u[global_inds[skel_inds[i]]]);
    }
}

// Obtain u_red = u(red), where red is the set of redundant DOFs.
//
// u (in): vector of size N^3
// data (in): factor data that contains the redundant indices
// skel_vec (out): u at the skeleton points
template <typename Scalar>
void GetRedundantVector(dmhm::Vector<Scalar>& u, FactorData<Scalar>& data,
                        dmhm::Vector<Scalar>& red_vec) {
    std::vector<int> global_inds = data.global_inds();
    std::vector<int> red_inds = data.redundant_set();
    red_vec.Resize(red_inds.size());
    for (size_t i = 0; i < red_inds.size(); ++i) {
        red_vec.Set(i, u[global_inds[red_inds[i]]]);
    }
}

// Copy the skeleton vector back into the global vector, u.
//
// u (out): vector of size N^3 that gets updated
// data (in): factor data containing skeleton indices
// skel_vec (in): u at the skeleton points, after updating
template <typename Scalar>
void CopySkeletonVector(dmhm::Vector<Scalar>& u, FactorData<Scalar>& data,
                        dmhm::Vector<Scalar>& skel_vec) {
    std::vector<int> global_inds = data.global_inds();
    std::vector<int> skel_inds = data.skeleton_set();
    assert(skel_vec.Size() == skel_inds.size());
    for (size_t i = 0; i < skel_inds.size(); ++i) {
        u.Set(u[global_inds[skel_inds[i]]], skel_vec[i]);
    }
}

// Copy the redundant vector back into the global vector, u.
//
// u (out): vector of size N^3 that gets updated
// data (in): factor data containing redundant indices
// red_vec (in): u at the redundant points, after updating
template <typename Scalar>
void CopyRedundantVector(dmhm::Vector<Scalar>& u, FactorData<Scalar>& data,
                         dmhm::Vector<Scalar>& red_vec) {
    std::vector<int> global_inds = data.global_inds();
    std::vector<int> red_inds = data.redundant_set();
    assert(red_vec.Size() == red_inds.size());
    for (size_t i = 0; i < red_inds.size(); ++i) {
        u.Set(u[global_inds[red_inds[i]]], red_vec[i]);
    }
}

// u_skel := alpha op(A) * u_red + u_skel
//
// op(A) is either A or A^H, dictated by the input 'adjoint'
// alpha is plus or minus 1, dictated by the input 'negative'.
// A is either data.W_mat() (default) or data.X_mat() (dictated by the input
// 'is_X' set to True)
// This function updates the entries of u at the skeleton points.
//
// u (in): vector or right-hand-side
// data (in): factorization data that stores op(A)
// u_skel (in): u at the skeleton points, i.e., the DOFs remaining.
//              This is not necessarily due to skeletonization.  u_skel
//              could correspond to the interior of faces in the Schur
//              factorization _or_ the skeleton points in skeletonization.
// u_red (in): u at the redundant points, i.e., the DOFs being eliminated.
// adjoint (in): if true, op(A) = A^H; otherwise, op(A) = A
// negative (in): if true, alpha = -1; otherwise, alpha = 1
// is_X (in): if true, A = data.X_mat(); otherwise, A = data.W_mat()
template <typename Scalar>
void UpdateSkeleton(dmhm::Vector<Scalar>& u, FactorData<Scalar>& data,
                    dmhm::Vector<Scalar>& u_skel, dmhm::Vector<Scalar>& u_red, 
                    bool adjoint, bool negative, bool is_X) {
    Scalar alpha = Scalar(1.0);
    if (negative) {
        alpha = Scalar(-1.0);
    }
    dmhm::Dense<Scalar>& A = data.W_mat();
    if (is_X) {
        A = data.X_mat();
    }
    if (adjoint) {
        dmhm::hmat_tools::AdjointMultiply(alpha, A, u_red, Scalar(1.0), u_skel);
    } else {
        dmhm::hmat_tools::Multiply(alpha, A, u_red, Scalar(1.0), u_skel);
    }
    CopySkeletonVector(u, data, u_skel);
}

// u_red := alpha op(A) * u_skel + u_red
//
// op(A) is either A or A^H, dictated by the input 'adjoint'
// alpha is plus or minus 1, dictated by the input 'negative'.
// A is either data.W_mat() (default) or data.X_mat() (dictated by the input
// 'is_X' set to True)
// This function updates the entries of u at the redundant points.
//
// u (in): vector or right-hand-side
// data (in): factorization data that stores op(A)
// u_skel (in): u at the skeleton points, i.e., the DOFs remaining.
//              This is not necessarily due to skeletonization.  u_skel
//              could correspond to the interior of faces in the Schur
//              factorization _or_ the skeleton points in skeletonization.
// u_red (in): u at the redundant points, i.e., the DOFs being eliminated.
// adjoint (in): if true, op(A) = A^H; otherwise, op(A) = A
// negative (in): if true, alpha = -1; otherwise, alpha = 1
// is_X (in): if true, A = data.X_mat(); otherwise, A = data.W_mat()
template <typename Scalar>
void UpdateRedundant(dmhm::Vector<Scalar>& u, FactorData<Scalar>& data,
                     dmhm::Vector<Scalar>& u_skel, dmhm::Vector<Scalar>& u_red, 
                     bool adjoint, bool negative, bool is_X) {
    Scalar alpha = Scalar(1.0);
    if (negative) {
        alpha = Scalar(-1.0);
    }
    dmhm::Dense<Scalar>& A = data.W_mat();
    if (is_X) {
        A = data.X_mat();
    }
    if (adjoint) {
        dmhm::hmat_tools::AdjointMultiply(alpha, A, u_skel, Scalar(1.0), u_red);
    } else {
        dmhm::hmat_tools::Multiply(alpha, A, u_skel, Scalar(1.0), u_red);
    }
    CopyRedundantVector(u, data, u_red);
}

// Apply A_{22} or A_{22}^{-1} to the redundant DOFs.
// 
// u (out): vector that gets updated 
// data (in): factorization data that contains the redundant indices, A_{22}, and A_{22}^{-1}
// inverse (in): if true, applies A_{22}^{-1}; otherwise, applies A_{22}
template <typename Scalar>
void ApplyA22(dmhm::Vector<Scalar>& u, FactorData<Scalar>& data, bool inverse) {
    dmhm::Vector<Scalar> u_red;
    GetRedundantVector(u, data, u_red);
    dmhm::Vector<Scalar> result(u_red.Size());
    if (inverse) {
        dmhm::hmat_tools::Multiply(Scalar(1.0), data.A_22_inv(), u_red, result);
    } else {
        dmhm::hmat_tools::Multiply(Scalar(1.0), data.A_22(), u_red, result);
    }
    CopyRedundantVector(u, data, result);
}

template <typename Scalar>
void HIFFactor<Scalar>::Apply(dmhm::Vector<Scalar>& u, bool inverse) {
    int num_levels = schur_level_data_.size();
    assert(num_levels == static_cast<int>(skel_level_data_.size()));
    assert(u.Size() == N_ * N_ * N_);
    
    for (int level = 0; level < num_levels - 1; ++level) {
        for (size_t j = 0; j < schur_level_data_[level].size(); ++j) {
            FactorData<Scalar>& data = schur_level_data_[level][j];
            dmhm::Vector<Scalar> u_skel, u_red;
            GetSkeletonVector(u, data, u_skel);
            GetRedundantVector(u, data, u_red);
            if (inverse) {
                // u_skel := - X^H * u_red + u_skel
                UpdateSkeleton(u, data, u_skel, u_red, true, true, true);
            } else {
                // u_red := X * u_skel + u_red
                UpdateRedundant(u, data, u_skel, u_red, false, false, true);
            }
        }
        for (size_t j = 0; j < skel_level_data_[level].size(); ++j) {
            FactorData<Scalar>& data = skel_level_data_[level][j];
            dmhm::Vector<Scalar> u_skel, u_red;
            GetSkeletonVector(u, data, u_skel);
            GetRedundantVector(u, data, u_red);
            if (inverse) {
                // u_red := -W^H * u_skel + u_red
                UpdateRedundant(u, data, u_skel, u_red, true, true, false);
                // u_skel := -X^H * u_red + u_skel
                UpdateSkeleton(u, data, u_skel, u_red, true, true, true);
            } else {
                // u_skel := W * u_red + u_skel
                UpdateSkeleton(u, data, u_skel, u_red, false, false, false);
                // u_red := X * u_skel + u_red
                UpdateRedundant(u, data, u_skel, u_red, false, false, true);
            }
        }
    }

    for (int level = 0; level < num_levels; ++level) {
        for (size_t j = 0; j < schur_level_data_[level].size(); ++j) {
            FactorData<Scalar>& data = schur_level_data_[level][j];
            // u_red := A_22 * u_red
            ApplyA22(u, data, inverse);
        }
        for (size_t j = 0; j < skel_level_data_[level].size(); ++j) {
            FactorData<Scalar>& data = skel_level_data_[level][j];
            // u_red := A_22 * u_red
            ApplyA22(u, data, inverse);
        }
    }

    for (int level = num_levels - 2; level >= 0; --level) {
        for (size_t j = 0; j < skel_level_data_[level].size(); ++j) {
            FactorData<Scalar>& data = skel_level_data_[level][j];
            dmhm::Vector<Scalar> u_skel, u_red;
            GetSkeletonVector(u, data, u_skel);
            GetRedundantVector(u, data, u_red);
            if (inverse) {
                // u_red := -X * u_skel + u_red
                UpdateRedundant(u, data, u_skel, u_red, false, true, true);
                // u_skel := -W * u_red + u_skel
                UpdateSkeleton(u, data, u_skel, u_red, false, true, false);
            } else {
                // u_skel := X * u_red + u_skel
                UpdateSkeleton(u, data, u_skel, u_red, true, false, true);
                // u_red := W * u_skel + u_red
                UpdateRedundant(u, data, u_skel, u_red, true, false, false);
            }
        }
        for (size_t j = 0; j < schur_level_data_[level].size(); ++j) {
            FactorData<Scalar>& data = schur_level_data_[level][j];
            dmhm::Vector<Scalar> u_skel, u_red;
            GetSkeletonVector(u, data, u_skel);
            GetRedundantVector(u, data, u_red);
            if (inverse) {
                // u_red := -X * u_skel + u_red
                UpdateRedundant(u, data, u_skel, u_red, false, true, false);
            } else {
                // u_skel := X * u_red + u_skel
                UpdateSkeleton(u, data, u_skel, u_red, true, false, true);
            }
        }
    }
}

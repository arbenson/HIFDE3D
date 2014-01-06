#include "Factor.hpp"
#include "data.hpp"
#include "dmhm/core/vector.hpp"

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
		;
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

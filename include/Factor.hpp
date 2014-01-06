#ifndef FACTOR_HPP_
#define FACTOR_HPP_

#include "data.hpp"
#include "dmhm/core/dense.hpp"
#include "dmhm/core/hmat_tools.hpp"
#include "dmhm/core/vector.hpp"
#include "NumTns.hpp"
#include "schur.hpp"
#include "dmhm/core/sparse.hpp"
#include "vec3t.hpp"

#include <vector>

#include <math.h>

template <typename Scalar>
class HIFFactor {
public:
    // TODO: better constructor and destructor
    HIFFactor() {}
    ~HIFFactor() {}

    // Initialize the factorization.  This function should be called
    // after the sparse matrix has been formatted and before Factor()
    // is called.
    void Initialize();

    // Form factorization of the matrix via HIF.
    // The matrix, number of discretiztaion points (N), width at lowest
    // level (P), and target accuracy (epsilon) must already be be provided.
    void Factor();

    // Convert a linear index to a tensor index. The tensor is of size
    // N x N x N, where N is one plus the number of discretization points per direction.
    Index3 Linear2TensorInd(int ind);

    // Convert a tensor index to a linear index. The tensor is of size
    // N x N x N, where N is one plus the number of discretization points per direction.
    int Tensor2LinearInd(Index3 ind);

    // Apply the factored matrix (or its inverse) to the vector u
    //
    // u (in): vector to which to apply the (inverse of the) matrix
    // apply_inverse (in): whether or not to apply the inverse
    void Apply(Vector<Scalar>& u, bool apply_inverse=false);

    void set_N(int N);
    int N();

    void set_P(int P);
    int P();

    void set_epsilon(double epsilon);
    double epsilon();

    std::vector< std::vector< FactorData<Scalar> > >& schur_level_data();
    std::vector< std::vector< FactorData<Scalar> > >& skel_level_data();

    // We assume that any (i, j, k) index with a i, j, or k = 0 is _zero_
    // This is the zero boundary conditions
    dmhm::Sparse<Scalar>& sp_matrix();


private:
    // Remove eliminated degrees of freedom.
    //
    // level (in): level at which to remove DOFs
    // is_skel (in): wheter or not the update is after skeletonization (after
    //               half level)
    void UpdateRemainingDOFs(int level, bool is_skel);

    // Eliminate DOFs interior to cells at a given level.
    //
    // cells_per_dir (in): Number of cells per direction.  There are
    //                     cells_per_dir ^ 3 cells on which to eliminate
    //                     interior DOFs.
    // level (in): level at which to eliminate DOFs
    void LevelFactorSchur(int cells_per_dir, int level);

    // Eliminate DOFs interior to cell faces via skeletonizations.
    //
    // cells_per_dir (in): Number of cells per direction.  There are
    //                     cells_per_dir ^ 3 cells on which to eliminate
    //                     interior DOFs.
    // level (in): level at which to eliminate DOFs
    void LevelFactorSkel(int cells_per_dir, int level);

    // Perform skeletonization on the DOFs corresponding to the
    // interior of a single face.
    //
    // cell_location (in): 3-tuple of cell location
    // Face (in): which face
    // level (in): which level of the algorithm
    // data (out): fills in all factorization data needed for this DOF set
    //             for solves after factorization completes
    // return value: True if and only if the face was available for skeletonization.
    //               For example, if cell_location = (0, 1, 2) and face = TOP, then
    //               the return value is false.
    bool Skeletonize(Index3 cell_location, Face face, int level,
                     FactorData<Scalar>& data);

    // Eliminate redundant DOFs via Schur complements after the ID has
    // completed.
    // 
    // data (in/out): Needs DOF data (global indices and skeleton/redundant indices)
    //                filled in.  After function completes, all data is filled in.
    void SchurAfterID(FactorData<Scalar>& data);

    // Update the global sparse matrix with the Schur complements from a
    // given level.  Also updates the remaining degrees of freedom.
    // This function should be called after computing all Schur complements
    // from either an integer level (just Schur complements) or half
    // level (skeletonization). 
    //
    // level (in): the level for which computation has completed and for which
    //             data will be used for the update
    // is_skel (in): wheter or not the update is after skeletonization (after
    //               half level)
    void UpdateMatrixAndDOFs(int level, bool is_skel);

    // For a given cell location at a given level, determine the indices of the
    // DOFs interior to the cell.  These DOFs are eliminated by a Schur
    // complement Also, determine the interaction of the interior DOFs.
    // The IndexData is filled with 
    //
    // cell_location (in): 3-tuple of cell location
    // level (in): level of the cell location
    // data (out): fills global indices, DOF set, and DOF set interactions.
    void InteriorCellIndexData(Index3 cell_location, int level, IndexData& data);
    
    // For a given cell location, level, and face, determine the indices of the
    // DOFs interior to the face and their interactions.  These DOFs are skeletonized.
    //
    // cell_location (in): 3-tuple of cell location
    // Face (in): which face
    // level (in): level of the cell location
    // data (out): fills global row and global column indices for skeletonization
    void InteriorFaceIndexData(Index3 cell_location, Face face, int level,
                               SkelIndexData& data);

    // For a given cell location, level, and face, determine the indices of the
    // DOFs interior to the face.
    //
    // cell_location (in): 3-tuple of cell location
    // Face (in): which face
    // level (in): level of the cell location
    // face_inds (out): adds indices of remaining face DOFs.
    void InteriorFaceDOFs(Index3 cell_location, Face face,
                          int level, std::vector<int>& face_inds);

    // Determine whether an index is on the interior of a cell.
    //
    // level (in): level of the partition
    // ind (in): index
    bool IsCellInterior(int level, Index3 ind);

    // Determine whether an index is on the interior at a face.
    //
    // level (in): level of the partition
    // ind (in): index
    bool IsFaceInterior(int level, Index3 ind);

    // Determine whether an index is on the interior at a given level.
    //
    // level (in): level of the partition
    // ind (in): index
    bool IsInterior(int level, int a);

    // Determine whether an index corresponds to a remaining DOF
    //
    // ind (in): 3-tuple index of DOF
    // return value: true if and only if the index corresponds to a remaining DOF
    bool IsRemainingDOF(Index3 ind);

    // DATA
    dmhm::Sparse<Scalar> sp_matrix_;
    int N_;
    int P_;
    double epsilon_;
    IntNumTns remaining_DOFs_;
    std::vector< std::vector< FactorData<Scalar> > > schur_level_data_;
    std::vector< std::vector< FactorData<Scalar> > > skel_level_data_;
};

#endif  // ifndef FACTOR_HPP_

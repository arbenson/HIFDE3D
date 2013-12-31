#ifndef INDEX_DATA_HPP_
#define INDEX_DATA_HPP_

#include <vector>

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

// For a given cell location at a given level, determine the indices of the
// DOFs interior to the cell.  These DOFs are eliminated by a Schur
// complement Also, determine the interaction of the interior DOFs.
//
// cell_location (in): 3-tuple of cell location
// W (in): width of the cell
// N (in): number of discretization points in each direction
// remaining_DOFs (in): 3-dimensional array of remaining degrees of freedom
// data (out): indexing data
// return value: 0 on failure, 1 on success
int InteriorCellIndexData(Index3 cell_location, int W, int N,
			  const IntNumTns& remaining_DOFs, IndexData& data);

// For a given cell location, level, and face, determine the indices of the
// DOFs interior to the face.  These DOFs are skeletonized.  Also, determine
// the interaction of the interior face DOFs.
//
// cell_location (in): 3-tuple of cell location
// Face (in): which face
// W (in): width of the cell
// N (in): number of discretization points in each direction
// remaining_DOFs (in): 3-dimensional array of remaining degrees of freedom
// data (out): indexing data
// return value: 0 on failure, 1 on success
int InteriorFaceIndexData(Index3 cell_location, Face face, int W, int N,
			  const IntNumTns& remaining_DOFs, IndexData& data);

#endif  // ifndef INDEX_DATA_HPP_

/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_SPARSE_MATRIX_HPP
#define DMHM_SPARSE_MATRIX_HPP 1

namespace dmhm {

// A simple Compressed Sparse Row (CSR) data structure
template<typename Scalar>
class Sparse
{
private:
    bool symmetric_;
    int height_, width_, nnz_;
    std::map<int,std::map<int,Scalar> > sparsemat_;

public:
    void Add( int i, int j, Scalar val );
    void Add( Vector<int> iidx, Vector<int> jidx, Dense<Scalar> vals );
    void Add( int i, Vector<int> jidx, Vector<Scalar> vals );
    void Add( Vector<int> iidx, int j, Vector<Scalar> vals );
    void Delete( int i, int j );
    void Delete( Vector<int> iidx, Vector<int> jidx );
    Scalar Find( int i, int j ) const;
    bool Check( int i, int j ) const;
    Dense<Scalar> Find( Vector<int> iidx, Vector<int> jidx );
    Vector<Scalar> Find( int i, Vector<int> jidx );
    Vector<Scalar> Find( Vector<int> iidx, int j );
    Vector<Scalar> FindRow( int i );
    Vector<Scalar> FindCol( int j );

    Scalar& operator[]( int i, int j );
    { return Find(i,j); }
    Dense<Scalar>& operator[]( Vector<int> iidx, Vector<int> jidx )
    { return Find(iidx,jidx); }
    Vector<Scalar>& operator[]( int i, Vector<int> jidx )
    { return Find(i,jidx); }
    Vector<Scalar>& operator[]( Vector<int> iidx, int j )
    { return Find(iidx,j); }

    void Clear();
    void Print( const std::string tag, std::ostream& os=std::cout ) const;
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar>
inline void
Sparse<Scalar>::Add( int i, int j, Scalar val )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Add");
#endif
    if( sparsemat_.find(i) == sparsemat_.end() )
    {
        sparsemat_[i] = new map<int, Scalar>;
        sparsemat_[i][j] = val;
    }
    else
    {
        map<int, Scalar> &irow = sparsemat_[i];
        if( irow.find(j) != irow.end() )
        {
#ifndef RELEASE
            throw std::logic_error("Add to a already exist position");
#endif
            irow[j] += val;
        }
        else
            irow[j] = val;
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::Add( int i, Vector<int> jidx, Vector<Scalar> vals )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Add");
#endif
    if( sparsemat_.find(i) == sparsemat_.end() )
    {
        sparsemat_[i] = new map<int, Scalar>;
        map<int, Scalar> &irow = sparsemat_[i];
        for( int iter=0; iter<jidx.Size(); ++iter )
		{
            int j = jidx.Get(iter);
            irow[j] = vals[iter];
        }
    }
    else
    {
        map<int, Scalar> &irow = sparsemat_[i];
        for( int iter=0; iter<jidx.Size(); ++iter )
		{
            int j = jidx.Get(iter);
            if( irow.find(j) != irow.end() )
            {
#ifndef RELEASE
                throw std::logic_error("Add to a already exist position");
#endif
                irow[j] += vals[iter];
            }
            else
                irow[j] = vals[iter];
        }
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::Add( int i, Vector<int> jidx, Vector<Scalar> vals )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Add");
#endif
    if( sparsemat_.find(i) == sparsemat_.end() )
    {
        sparsemat_[i] = new map<int, Scalar>;
        map<int, Scalar> &irow = sparsemat_[i];
        for( int iter=0; iter<jidx.Size(); ++iter )
		{
            int j = jidx.Get(iter);
            irow[j] = vals[iter];
        }
    }
    else
    {
        map<int, Scalar> &irow = sparsemat_[i];
        for( int iter=0; iter<jidx.Size(); ++iter )
		{
            int j = jidx.Get(iter);
            if( irow.find(j) != irow.end() )
            {
#ifndef RELEASE
                throw std::logic_error("Add to a already exist position");
#endif
                irow[j] += vals[iter];
            }
            else
                irow[j] = vals[iter];
        }
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::Clear()
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Clear");
#endif
    symmetric = false;
    height = 0;
    width = 0;
    nonzeros.Clear();
    columnIndices.Clear();
    rowOffsets.Clear();
}

template<typename Scalar>
inline void
Sparse<Scalar>::Print( const std::string tag, std::ostream& os ) const
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Print");
#endif
    if( symmetric )
        os << tag << "(symmetric)\n";
    else
        os << tag << "\n";

    for( int i=0; i<height; ++i )
    {
        const int numCols = rowOffsets[i+1]-rowOffsets[i];
        const int rowOffset = rowOffsets[i];
        for( int k=0; k<numCols; ++k )
        {
            const int j = columnIndices[rowOffset+k];
            const Scalar alpha = nonzeros[rowOffset+k];
            os << i << " " << j << " " << WrapScalar(alpha) << "\n";
        }
    }
    os << std::endl;
}

} // namespace dmhm

#endif // ifndef DMHM_SPARSE_MATRIX_HPP

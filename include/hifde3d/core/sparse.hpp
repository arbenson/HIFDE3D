/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (HIFDE3D) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef HIFDE3D_SPARSE_MATRIX_HPP
#define HIFDE3D_SPARSE_MATRIX_HPP 1

namespace hifde3d {

// A simple Compressed Sparse Row (CSR) data structure
template<typename Scalar>
class Sparse
{
private:
    int height_, width_, nnz_;
    std::map<int,std::map<int,Scalar> > sparsemat_;

public:
    Sparse();
    Sparse( int height, int width );

    // This function should not be used, dengerous.
    Sparse( const Sparse<Scalar>& x );
    ~Sparse();

    int Height() const{ return height_; }
    int Width() const{ return width_; }
    int NonZeros() const{ return nnz_; }

    // This function should not be used, dengerous.
    std::map<int,std::map<int,Scalar> > SparseMat() const
    { return sparsemat_; }

    void Add( int i, int j, Scalar val );
    void Add( Vector<int>& iidx, Vector<int>& jidx, Dense<Scalar>& vals );
    void Add( int i, Vector<int>& jidx, Vector<Scalar>& vals );
    void Add( Vector<int>& iidx, int j, Vector<Scalar>& vals );
    void Delete( int i, int j );
    void DeleteRow( int i );
    void DeleteCol( int j );
    void DeleteRow( Vector<int>& iidx );
    void DeleteCol( Vector<int>& jidx );
    void Delete( Vector<int>& iidx, Vector<int>& jidx );
    Scalar Find( int i, int j );
    bool Check( int i, int j ) const;
    void Find
    ( Vector<int>& iidx, Vector<int>& jidx, Dense<Scalar>& res );
    void Find( int i, Vector<int>& jidx, Vector<Scalar>& res );
    void Find( Vector<int>& iidx, int j, Vector<Scalar>& res );
    void FindRow( int i, Vector<Scalar>& res ) const;
    void FindCol( int j, Vector<Scalar>& res ) const;

    /*
    Scalar& operator()( int i, int j )
    { return Find(i,j); }
    Dense<Scalar>& operator()( Vector<int>& iidx, Vector<int>& jidx )
    { return Find(iidx,jidx); }
    Vector<Scalar>& operator()( int i, Vector<int>& jidx )
    { return Find(i,jidx); }
    Vector<Scalar>& operator()( Vector<int>& iidx, int j )
    { return Find(iidx,j); }
    */

    void Clear();
    void Print( const std::string tag, std::ostream& os=std::cout ) const;
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//
template<typename Scalar>
inline
Sparse<Scalar>::Sparse()
: height_(0), width_(0), nnz_(0)
{}

template<typename Scalar>
inline
Sparse<Scalar>::Sparse( int height, int width )
: height_(height), width_(width), nnz_(0)
{}

template<typename Scalar>
inline
Sparse<Scalar>::Sparse( const Sparse<Scalar>& x )
: height_(x.Height), width_(x.Width), nnz_(x.NonZeros)
{
    sparsemat_ = x.SparseMat();
}

template<typename Scalar>
inline
Sparse<Scalar>::~Sparse()
{
    Clear();
}

template<typename Scalar>
inline void
Sparse<Scalar>::Add( int i, int j, Scalar val )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Add");
#endif
    if( sparsemat_.find(i) == sparsemat_.end() )
    {
        sparsemat_[i] = new std::map<int, Scalar>;
        sparsemat_[i][j] = val;
        nnz_++;
    }
    else
    {
        std::map<int, Scalar> &irow = sparsemat_[i];
        if( irow.find(j) != irow.end() )
        {
            irow[j] += val;
        }
        else
        {
            irow[j] = val;
            nnz_++;
        }
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::Add( int i, Vector<int>& jidx, Vector<Scalar>& vals )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Add");
#endif
    if( sparsemat_.find(i) == sparsemat_.end() )
    {
	std::map<int, Scalar>& irow = sparsemat_[i];
        for( int iter=0; iter<jidx.Size(); ++iter )
		{
            int j = jidx.Get(iter);
            irow[j] = vals.Get(iter);
            nnz_++;
        }
    }
    else
    {
        std::map<int, Scalar> &irow = sparsemat_[i];
        for( int iter=0; iter<jidx.Size(); ++iter )
		{
            int j = jidx.Get(iter);
            if( irow.find(j) != irow.end() )
            {
                irow[j] += vals.Get(iter);
            }
            else
            {
                irow[j] = vals.Get(iter);
                nnz_++;
            }
        }
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::Add( Vector<int>& iidx, int j, Vector<Scalar>& vals )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Add");
#endif
    for( int iteri=0; iteri<iidx.Size(); ++iteri )
    {
        int i = iidx.Get(iteri);
        if( sparsemat_.find(i) == sparsemat_.end() )
        {
            sparsemat_[i] = new std::map<int, Scalar>;
            std::map<int, Scalar> &irow = sparsemat_[i];
            irow[j] = vals.Get(iteri);
            nnz_++;
        }
        else
        {
            std::map<int, Scalar> &irow = sparsemat_[i];
            if( irow.find(j) != irow.end() )
            {
                irow[j] += vals.Get(iteri);
            }
            else
            {
                irow[j] = vals.Get(iteri);
                nnz_++;
            }
        }
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::Add( Vector<int>& iidx, Vector<int>& jidx,
                     Dense<Scalar>& vals )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Add");
#endif
    for( int iteri=0; iteri<iidx.Size(); ++iteri )
    {
        int i = iidx.Get(iteri);
        if( sparsemat_.find(i) == sparsemat_.end() )
        {
            sparsemat_[i] = new std::map<int, Scalar>;
            std::map<int, Scalar> &irow = sparsemat_[i];
            for( int iter=0; iter<jidx.Size(); ++iter )
	    	{
                int j = jidx.Get(iter);
                irow[j] = vals.Get(iter);
                nnz_++;
            }
        }
        else
        {
            std::map<int, Scalar> &irow = sparsemat_[i];
            for( int iter=0; iter<jidx.Size(); ++iter )
	    	{
                int j = jidx.Get(iter);
                if( irow.find(j) != irow.end() )
                {
                    irow[j] += vals.Get(iter);
                }
                else
                {
                    irow[j] = vals.Get(iter);
                    nnz_++;
                }
            }
        }
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::Delete( int i, int j )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Delete");
#endif
    if( sparsemat_.find(i) != sparsemat_.end() )
    {
        std::map<int, Scalar> &irow = sparsemat_[i];
        if( irow.find(j) != irow.end() )
        {
            irow.erase(j);
            nnz_--;
        }
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::DeleteRow( int i )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::DeleteRow");
#endif
    if( sparsemat_.find(i) != sparsemat_.end() )
    {
        std::map<int, Scalar> &irow = sparsemat_[i];
        nnz_ -= irow.size();
        irow.clear();
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::DeleteCol( int j )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::DeleteCol");
#endif
    typename std::map<int,std::map<int,Scalar> >::iterator it;
    for( it=sparsemat_.begin(); it!=sparsemat_.end(); ++it )
    {
        std::map<int, Scalar> &irow = it->second;
        if( irow.find(j) != irow.end() )
        {
            irow.erase(j);
            nnz_--;
        }
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::DeleteRow( Vector<int>& iidx )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::DeleteRow");
#endif
    for( int iter=0; iter<iidx.Size(); ++iter )
    {
        int i = iidx.Get(iter);
        DeleteRow(i);
    }
}

//DeleteCol(vector) can be rewrite, which could
//reduce the complexity by a factor of constant.
template<typename Scalar>
inline void
Sparse<Scalar>::DeleteCol( Vector<int>& jidx )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::DeleteCol");
#endif
    typename std::map<int,std::map<int,Scalar> >::iterator it;
    for( it=sparsemat_.begin(); it!=sparsemat_.end(); ++it )
    {
	std::map<int, Scalar> &irow = it->second;
	// std::cout << it->first << std::endl;
	for (int iter=0; iter<jidx.Size(); ++iter) {
	    int j = jidx.Get(iter);
	    if( irow.find(j) != irow.end() ) {
		    irow.erase(j);
		    nnz_--;
	    }
	}
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::Delete( Vector<int>& iidx, Vector<int>& jidx )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Delete");
#endif
    for( int iteri=0; iteri<iidx.Size(); ++iteri )
    {
        int i = iidx.Get(iteri);
        if( sparsemat_.find(i) != sparsemat_.end() )
        {
            std::map<int, Scalar> &irow = sparsemat_[i];
            for( int iter=0; iter<jidx.Size(); ++iter )
	    	{
                int j = jidx.Get(iter);
                if( irow.find(j) != irow.end() )
                {
                    irow.erase(j);
                    nnz_--;
                }
            }
        }
    }
}

template<typename Scalar>
inline Scalar
Sparse<Scalar>::Find( int i, int j )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Find");
#endif
    if( sparsemat_.find(i) != sparsemat_.end() )
    {
        std::map<int, Scalar> irow = sparsemat_[i];
        if( irow.find(j) != irow.end() )
            return irow[j];
        else
        {
#ifndef RELEASE
            throw std::logic_error("Position does not exist");
#endif
        }
    }
    else
    {
#ifndef RELEASE
        throw std::logic_error("Position does not exist");
#endif
    }
}

template<typename Scalar>
inline bool
Sparse<Scalar>::Check( int i, int j ) const
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Check");
#endif
    if( sparsemat_.find(i) != sparsemat_.end() )
    {
        std::map<int, Scalar>& irow = sparsemat_[i];
        if( irow.find(j) != irow.end() )
            return true;
        else
            return false;
    }
    else
        return false;
}

template<typename Scalar>
inline void
Sparse<Scalar>::Find
( Vector<int>& iidx, Vector<int>& jidx, Dense<Scalar>& D )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Find");
#endif
    for( int iteri=0; iteri<iidx.Size(); ++iteri )
    {
        int i = iidx.Get(iteri);
        if( sparsemat_.find(i) != sparsemat_.end() )
        {
	    /*
	    std::cout << "i: " << i << std::endl;
	    */
	    std::map<int, Scalar>& irow = sparsemat_[i];
            for( int iterj=0; iterj<jidx.Size(); ++iterj )
            {
                int j = jidx.Get(iterj);
                if( irow.find(j) != irow.end() ) {
		    /*
		    if (i == 2081) {
			std::cout << "found: " << i << " " << j << " " << irow[j] << std::endl;
		    }
		    */
                    D.Set(iteri,iterj,irow[j]);
		}
                else {
		    /*
		    if (i == 2081) {
			std::cout << "could not find: " << i << " " << j << std::endl;
		    }
		    */
                    D.Set(iteri,iterj,(Scalar)0);
		}
            }
        }
        else
        {
	    std::cout << "found zero row!!" << std::endl;
            for( int iterj=0; iterj<jidx.Size(); ++ iterj )
                D.Set(iteri,iterj,(Scalar)0);
        }
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::Find
( int i, Vector<int>& jidx, Vector<Scalar>& vec )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Find");
#endif
    if( sparsemat_.find(i) != sparsemat_.end() )
    {
        std::map<int, Scalar> &irow = sparsemat_[i];
        for( int iterj=0; iterj<jidx.Size(); ++iterj )
        {
            int j = jidx.Get(iterj);
            if( irow.find(j) != irow.end() )
                vec.Set(iterj,irow[j]);
            else
                vec.Set(iterj,(Scalar)0);
        }
    }
    else
    {
        for( int iterj=0; iterj<jidx.Size(); ++ iterj )
            vec.Set(iterj,(Scalar)0);
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::Find
( Vector<int>& iidx, int j, Vector<Scalar>& vec )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Find");
#endif
    for( int iteri=0; iteri<iidx.Size(); ++iteri )
    {
        int i = iidx.Get(iteri);
        if( sparsemat_.find(i) != sparsemat_.end() )
        {
            std::map<int, Scalar> &irow = sparsemat_[i];
            if( irow.find(j) != irow.end() )
                vec.Set(iteri,irow[j]);
            else
                vec.Set(iteri,(Scalar)0);
        }
        else
            vec.Set(iteri,(Scalar)0);
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::FindRow( int i, Vector<Scalar>& vec ) const
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::FindRow");
#endif
    if( sparsemat_.find(i) != sparsemat_.end() )
    {
        std::map<int,Scalar> &irow = sparsemat_[i];
        typename std::map<int,Scalar>::iterator it;
        for( it=irow.begin(); it!=irow.end(); ++it )
            vec.Set(it->first,it->second);
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::FindCol( int j, Vector<Scalar>& vec ) const
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::FindCol");
#endif
    typename std::map<int,std::map<int,Scalar> >::iterator it;
    for( it=sparsemat_.begin(); it!=sparsemat_.end(); ++it )
    {
        std::map<int, Scalar> &irow = it->second;
        if( irow.find(j) != irow.end() )
            vec.Set(it->first,irow[j]);
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::Clear()
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Clear");
#endif
    height_ = 0;
    width_ = 0;
    nnz_ = 0;
    sparsemat_.clear();
}

template<typename Scalar>
inline void
Sparse<Scalar>::Print( const std::string tag, std::ostream& os ) const
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Print");
#endif
    os << tag << "\n";

    typename std::map<int,std::map<int,Scalar> >::iterator it;
    for( it=sparsemat_.begin(); it!=sparsemat_.end(); ++it )
    {
        std::map<int, Scalar> &irow = it->second;
        typename std::map<int, Scalar>::iterator itinner;
        for( itinner=irow.begin(); itinner!=irow.end(); ++itinner )
            os << it->first << " " << itinner->first << " "
               << WrapScalar(itinner->second) << "\n";
    }
    os << std::endl;
}

} // namespace hifde3d

#endif // ifndef HIFDE3D_SPARSE_MATRIX_HPP

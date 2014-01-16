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

// A simple Compressed Sparse Col (CSR) data structure
template<typename Scalar>
class Sparse
{
private:
    int height_, width_;
    std::map<int,std::map<int,Scalar> > sparsemat_;

public:
    Sparse();
    Sparse( int height, int width );
    ~Sparse();

    int Height() const{ return height_; }
    int Width() const{ return width_; }
    int NonZeros() const;

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
    void FindRow( int i, Vector<Scalar>& res) const;
    void FindRow( int i, Vector<Scalar>& res, Vector<int>& inds);
    void FindCol( int j, Vector<Scalar>& res ) const;

    void Clear();
    void Print( const std::string tag, std::ostream& os=std::cout ) const;
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//
template<typename Scalar>
inline
Sparse<Scalar>::Sparse()
: height_(0), width_(0)
{}

template<typename Scalar>
inline
Sparse<Scalar>::Sparse( int height, int width )
: height_(height), width_(width)
{}

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
    sparsemat_[j][i] += val;
}

template<typename Scalar>
inline void
Sparse<Scalar>::Add( Vector<int>& iidx, int j, Vector<Scalar>& vals )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Add");
#endif
    std::map<int, Scalar>& jcol = sparsemat_[j];
    for( int iter=0; iter<iidx.Size(); ++iter )
	{
        int i = iidx.Get(iter);
        jcol[i] += vals.Get(iter);
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::Add( int i, Vector<int>& jidx, Vector<Scalar>& vals )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::Add");
#endif
    for( int iterj=0; iterj<jidx.Size(); ++iterj )
    {
        int j = jidx.Get(iterj);
        sparsemat_[j][i] += vals.Get(iterj);
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
    for( int iterj=0; iterj<jidx.Size(); ++iterj )
    {
        int j = jidx.Get(iterj);
        std::map<int, Scalar> &jcol = sparsemat_[j];
        for( int iter=0; iter<iidx.Size(); ++iter )
	    {
            int i = jidx.Get(iter);
            jcol[i] += vals.Get(iter);
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
    std::map<int, Scalar> &jcol = sparsemat_[j];
    jcol.erase(i);
}

template<typename Scalar>
inline void
Sparse<Scalar>::DeleteCol( int j )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::DeleteCol");
#endif
        sparsemat_[j].clear();
}

template<typename Scalar>
inline void
Sparse<Scalar>::DeleteRow( int i )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::DeleteRow");
#endif
    typename std::map<int,std::map<int,Scalar> >::iterator it;
    for( it=sparsemat_.begin(); it!=sparsemat_.end(); ++it )
    {
        std::map<int, Scalar> &jcol = it->second;
        jcol.erase(i);
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::DeleteCol( Vector<int>& jidx )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::DeleteCol");
#endif
    for( int iter=0; iter<jidx.Size(); ++iter )
    {
        int j = jidx.Get(iter);
        sparsemat_[j].clear();
    }
}

//DeleteCol(vector) can be rewrite, which could
//reduce the complexity by a factor of constant.
template<typename Scalar>
inline void
Sparse<Scalar>::DeleteRow( Vector<int>& iidx )
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::DeleteRow");
#endif
    typename std::map<int,std::map<int,Scalar> >::iterator it;
    for( it=sparsemat_.begin(); it!=sparsemat_.end(); ++it )
    {
	    std::map<int, Scalar> &jcol = it->second;
	    for (int iter=0; iter<iidx.Size(); ++iter)
        {
	        int i = iidx.Get(iter);
		    jcol.erase(i);
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
    for( int iterj=0; iterj<jidx.Size(); ++iterj )
    {
        int j = jidx.Get(iterj);
        std::map<int, Scalar> &jcol = sparsemat_[j];
        for( int iter=0; iter<iidx.Size(); ++iter )
	    {
            int i = iidx.Get(iter);
            jcol.erase(i);
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
    if( sparsemat_.find(j) != sparsemat_.end() )
    {
        std::map<int, Scalar> jcol = sparsemat_[j];
        if( jcol.find(i) != jcol.end() )
            return jcol[i];
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
    if( sparsemat_.find(j) != sparsemat_.end() )
    {
        std::map<int, Scalar>& jcol = sparsemat_[j];
        if( jcol.find(i) != jcol.end() )
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
    for( int iterj=0; iterj<jidx.Size(); ++iterj )
    {
        int j = jidx.Get(iterj);
        if( sparsemat_.find(j) != sparsemat_.end() )
        {
	        std::map<int, Scalar>& jcol = sparsemat_[j];
            for( int iteri=0; iteri<iidx.Size(); ++iteri )
            {
                int i = iidx.Get(iteri);
                if( jcol.find(i) != jcol.end() ) {
                    D.Set(iteri,iterj,jcol[i]);
	         	}
                else {
                    D.Set(iteri,iterj,(Scalar)0);
		        }
            }
        }
        else
        {
            for( int iteri=0; iteri<iidx.Size(); ++ iteri )
                D.Set(iteri,iterj,(Scalar)0);
        }
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
    if( sparsemat_.find(j) != sparsemat_.end() )
    {
        std::map<int, Scalar> &jcol = sparsemat_[j];
        for( int iteri=0; iteri<iidx.Size(); ++iteri )
        {
            int i = iidx.Get(iteri);
            if( jcol.find(i) != jcol.end() )
                vec.Set(iteri,jcol[i]);
            else
                vec.Set(iteri,(Scalar)0);
        }
    }
    else
    {
        for( int iteri=0; iteri<iidx.Size(); ++ iteri )
            vec.Set(iteri,(Scalar)0);
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
    for( int iterj=0; iterj<jidx.Size(); ++iterj )
    {
        int j = jidx.Get(iterj);
        if( sparsemat_.find(j) != sparsemat_.end() )
        {
            std::map<int, Scalar> &jcol = sparsemat_[j];
            if( jcol.find(i) != jcol.end() )
                vec.Set(iterj,jcol[i]);
            else
                vec.Set(iterj,(Scalar)0);
        }
        else
            vec.Set(iterj,(Scalar)0);
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::FindCol( int j, Vector<Scalar>& vec ) const
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::FindCol");
#endif
    vec.Resize(height_);
    if( sparsemat_.find(j) != sparsemat_.end() )
    {
        std::map<int,Scalar> &jcol = sparsemat_[j];
        typename std::map<int,Scalar>::iterator it;
        for( it=jcol.begin(); it!=jcol.end(); ++it )
            vec.Set(it->first,it->second);
    }
}

template<typename Scalar>
inline void
Sparse<Scalar>::FindRow( int i, Vector<Scalar>& vec ) const
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::FindRow");
#endif
    vec.Resize(width_);
    typename std::map<int,std::map<int,Scalar> >::iterator it;
    for( it=sparsemat_.begin(); it!=sparsemat_.end(); ++it )
    {
        std::map<int, Scalar> &jcol = it->second;
        if( jcol.find(i) != jcol.end() )
            vec.Set(it->first,jcol[i]);
    }
}

template<typename Scalar>
inline int
Sparse<Scalar>::NonZeros() const
{
#ifndef RELEASE
    CallStackEntry entry("Sparse::NonZeros");
#endif
    int nnz = 0;
    typename std::map<int,std::map<int,Scalar> >::iterator it;
    for( it=sparsemat_.begin(); it!=sparsemat_.end(); ++it )
    {
        std::map<int, Scalar> &jcol = it->second;
        nnz += jcol.size();
    }
    return nnz;
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
        std::map<int, Scalar> &jcol = it->second;
        typename std::map<int, Scalar>::iterator itinner;
        for( itinner=jcol.begin(); itinner!=jcol.end(); ++itinner )
            os << itinner->first << " " << it->first << " "
               << WrapScalar(itinner->second) << "\n";
    }
    os << std::endl;
}

} // namespace hifde3d

#endif // ifndef HIFDE3D_SPARSE_MATRIX_HPP

/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (HIFDE3D) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef HIFDE3D_DENSE_HPP
#define HIFDE3D_DENSE_HPP 1

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace hifde3d {

enum MatrixType { GENERAL, SYMMETRIC /*, HERMITIAN*/ };

// A basic dense matrix representation that is used for storing blocks
// whose sources and targets are too close to represent as low rank
template<typename Scalar>
class Dense
{
private:
    /*
     * Private member data
     */
    int height_, width_;
    int ldim_; // leading dimension of matrix
    bool viewing_;
    bool lockedView_;
    std::vector<Scalar> memory_;
    Scalar* buffer_;
    const Scalar* lockedBuffer_;
    MatrixType type_;

public:
    /*
     * Public non-static member functions
     */
    Dense( MatrixType type=GENERAL );
    Dense( int height, int width, MatrixType type=GENERAL );
    Dense( int height, int width, int ldim, MatrixType type=GENERAL );
    Dense
    ( Scalar* buffer, int height, int width, int ldim,
      MatrixType type=GENERAL );
    Dense
    ( const Scalar* lockedBuffer, int height, int width, int ldim,
      MatrixType type=GENERAL );
    ~Dense();

    void SetType( MatrixType type );
    MatrixType Type() const;
    bool General() const;
    bool Symmetric() const;
    //bool Hermitian() const;

    int Height() const;
    int Width() const;
    int Size() const;
    int LDim() const;
    void Resize( int height, int width );
    void Resize( int height, int width, int ldim );
    void EraseCols( int first, int last );
    void EraseRows( int first, int last );
    void Erase( int colfirst, int collast, int rowfirst, int rowlast );
    void Clear();

    bool IsEmpty() const;

    void Set( int i, int j, Scalar value );
    Scalar Get( int i, int j ) const;
    void Print( const std::string tag, std::ostream& os=std::cout ) const;
    void Print2( const std::string tag, std::ostream& os=std::cout ) const;

    Scalar* Buffer( int i=0, int j=0 );
    const Scalar* LockedBuffer( int i=0, int j=0 ) const;

    void View( Dense<Scalar>& A );
    void View( Dense<Scalar>& A, int i, int j, int height, int width );

    void LockedView( const Dense<Scalar>& A );
    void LockedView
    ( const Dense<Scalar>& A, int i, int j, int height, int width );
    void Init( );
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename Scalar>
inline
Dense<Scalar>::Dense( MatrixType type )
: height_(0), width_(0), ldim_(1),
  viewing_(false), lockedView_(false),
  memory_(), buffer_(0), lockedBuffer_(0),
  type_(type)
{
#ifdef MEMORY_INFO
    AddToMemoryCount( memory_.size()*sizeof(Scalar) );
#endif
}

template<typename Scalar>
inline
Dense<Scalar>::Dense( int height, int width, MatrixType type )
: height_(height), width_(width), ldim_(std::max(height,1)),
  viewing_(false), lockedView_(false),
  memory_(ldim_*width_), buffer_(&memory_[0]), lockedBuffer_(0),
  type_(type)
{
#ifndef RELEASE
    CallStackEntry entry("Dense::Dense");
    if( height < 0 || width < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( type == SYMMETRIC && height != width )
        throw std::logic_error("Symmetric matrices must be square");
#endif
#ifdef MEMORY_INFO
    AddToMemoryCount( memory_.size()*sizeof(Scalar) );
#endif
}

template<typename Scalar>
inline
Dense<Scalar>::Dense( int height, int width, int ldim, MatrixType type )
: height_(height), width_(width), ldim_(ldim),
  viewing_(false), lockedView_(false),
  memory_(ldim_*width_), buffer_(&memory_[0]), lockedBuffer_(0),
  type_(type)
{
#ifndef RELEASE
    CallStackEntry entry("Dense::Dense");
    if( height < 0 || width < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( type == SYMMETRIC && height != width )
        throw std::logic_error("Symmetric matrices must be square");
    if( ldim <= 0 )
        throw std::logic_error("Leading dimensions must be positive");
#endif
#ifdef MEMORY_INFO
    AddToMemoryCount( memory_.size()*sizeof(Scalar) );
#endif
}

template<typename Scalar>
inline
Dense<Scalar>::Dense
( Scalar* buffer, int height, int width, int ldim, MatrixType type )
: height_(height), width_(width), ldim_(ldim),
  viewing_(true), lockedView_(false),
  memory_(), buffer_(buffer), lockedBuffer_(0),
  type_(type)
{
#ifndef RELEASE
    CallStackEntry entry("Dense::Dense");
    if( height < 0 || width < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( type == SYMMETRIC && height != width )
        throw std::logic_error("Symmetric matrices must be square");
    if( ldim <= 0 )
        throw std::logic_error("Leading dimensions must be positive");
#endif
}

template<typename Scalar>
inline
Dense<Scalar>::Dense
( const Scalar* lockedBuffer, int height, int width, int ldim, MatrixType type )
: height_(height), width_(width), ldim_(ldim),
  viewing_(true), lockedView_(true),
  memory_(), buffer_(0), lockedBuffer_(lockedBuffer),
  type_(type)
{
#ifndef RELEASE
    CallStackEntry entry("Dense::Dense");
    if( height < 0 || width < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( type == SYMMETRIC && height != width )
        throw std::logic_error("Symmetric matrices must be square");
    if( ldim <= 0 )
        throw std::logic_error("Leading dimensions must be positive");
#endif
}

template<typename Scalar>
inline
Dense<Scalar>::~Dense()
{ Clear(); }

template<typename Scalar>
inline void
Dense<Scalar>::SetType( MatrixType type )
{
#ifndef RELEASE
    CallStackEntry entry("Dense::SetType");
    if( type == SYMMETRIC && height_ != width_ )
        throw std::logic_error("Symmetric matrices must be square");
#endif
    type_ = type;
}

template<typename Scalar>
inline hifde3d::MatrixType
Dense<Scalar>::Type() const
{ return type_; }

template<typename Scalar>
inline bool
Dense<Scalar>::General() const
{ return type_ == GENERAL; }

template<typename Scalar>
inline bool
Dense<Scalar>::Symmetric() const
{ return type_ == SYMMETRIC; }

/*
template<typename Scalar>
inline bool
Dense<Scalar>::Hermitian() const
{
    for( int i=0; i<height_; ++i )
        for( int j=i+1; j<width_; ++j )
            if( std::abs(std::abs(Get(i,j))-std::abs(Get(j,i)))>1e-6 )
                return false;
    return true;
    //return type_ == HERMITIAN;
}
*/

template<typename Scalar>
inline int
Dense<Scalar>::Height() const
{ return height_; }

template<typename Scalar>
inline int
Dense<Scalar>::Width() const
{ return width_; }

template<typename Scalar>
inline int
Dense<Scalar>::Size() const
{ return memory_.size(); }

template<typename Scalar>
inline int
Dense<Scalar>::LDim() const
{ return ldim_; }

template<typename Scalar>
inline void
Dense<Scalar>::Resize( int height, int width )
{
#ifndef RELEASE
    CallStackEntry entry("Dense::Resize");
    if( viewing_ )
        throw std::logic_error("Cannot resize views");
    if( height < 0 || width < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( type_ == SYMMETRIC && height != width )
        throw std::logic_error("Destroyed symmetry of symmetric matrix");
#endif
#ifdef MEMORY_INFO
    AddToMemoryCount( -(double)memory_.size()*sizeof(Scalar) );
#endif
    if( height > ldim_ )
    {
        // We cannot trivially preserve the old contents
        ldim_ = std::max( height, 1 );
    }
    height_ = height;
    width_ = width;
    memory_.resize( ldim_*width );
#ifdef MEMORY_INFO
    AddToMemoryCount( memory_.size()*sizeof(Scalar) );
#endif
    buffer_ = &memory_[0];
}

template<typename Scalar>
inline void
Dense<Scalar>::Resize( int height, int width, int ldim )
{
#ifndef RELEASE
    CallStackEntry entry("Dense::Resize");
    if( viewing_ )
        throw std::logic_error("Cannot resize views");
    if( height < 0 || width < 0 )
        throw std::logic_error("Height and width must be non-negative");
    if( ldim < height || ldim < 0 )
        throw std::logic_error("LDim must be positive and >= the height");
    if( type_ == SYMMETRIC && height != width )
        throw std::logic_error("Destroyed symmetry of symmetric matrix");
#endif
#ifdef MEMORY_INFO
    AddToMemoryCount( -(double)memory_.size()*sizeof(Scalar) );
#endif
    height_ = height;
    width_ = width;
    ldim_ = ldim;
    memory_.resize( ldim*width );
#ifdef MEMORY_INFO
    AddToMemoryCount( memory_.size()*sizeof(Scalar) );
#endif
    buffer_ = &memory_[0];
}

template<typename Scalar>
inline void
Dense<Scalar>::EraseCols( int first, int last )
{
#ifndef RELEASE
    CallStackEntry entry("Dense::EraseCols");
    if( viewing_ )
        throw std::logic_error("Cannot erase views");
    if( first < 0 || last >= width_ )
        throw std::logic_error("First and last must be in the range of matrix");
    if( type_ == SYMMETRIC )
        throw std::logic_error("Destroyed symmetry of symmetric matrix");
#endif
#ifdef MEMORY_INFO
        AddToMemoryCount( -(double)memory_.size()*sizeof(Scalar) );
#endif
    if( first <= last )
    {
        width_ = width_-last+first-1;
        memory_.erase
        ( memory_.begin()+first*ldim_, memory_.begin()+(last+1)*ldim_ );
        buffer_ = &memory_[0];
    }
#ifdef MEMORY_INFO
        AddToMemoryCount( (double)memory_.size()*sizeof(Scalar) );
#endif
}

template<typename Scalar>
inline void
Dense<Scalar>::EraseRows( int first, int last )
{
#ifndef RELEASE
    CallStackEntry entry("Dense::EraseRows");
    if( viewing_ )
        throw std::logic_error("Cannot erase views");
    if( first < 0 || last >= height_ )
        throw std::logic_error("First and last must be in the range of matrix");
    if( type_ == SYMMETRIC )
        throw std::logic_error("Destroyed symmetry of symmetric matrix");
#endif
#ifdef MEMORY_INFO
            AddToMemoryCount( -(double)memory_.size()*sizeof(Scalar) );
#endif
    if( first <= last )
    {
        height_ = height_-last+first-1;
        for( int i=width_-1; i>=0; --i )
        {
            memory_.erase
            ( memory_.begin()+i*ldim_+first, memory_.begin()+i*ldim_+last+1 );
        }
        buffer_ = &memory_[0];
        ldim_ = std::max( ldim_-last+first-1, 1 );
    }
#ifdef MEMORY_INFO
            AddToMemoryCount( (double)memory_.size()*sizeof(Scalar) );
#endif
}

template<typename Scalar>
inline void
Dense<Scalar>::Erase
( int colfirst, int collast, int rowfirst, int rowlast )
{
    MatrixType typetmp = type_;
#ifndef RELEASE
    CallStackEntry entry("Dense::Erase");
    if( viewing_ )
        throw std::logic_error("Cannot erase views");
    if( rowfirst < 0 || rowlast >= height_ ||
        colfirst<0 || collast >= width_ )
        throw std::logic_error("First and last must be in the range of matrix");
    if( type_ == SYMMETRIC && ( colfirst != rowfirst || collast != rowlast ) )
        throw std::logic_error("Destroyed symmetry of symmetric matrix");
    if( type_ == SYMMETRIC )
        type_ = GENERAL;
#endif
    EraseCols( colfirst, collast );
    EraseRows( rowfirst, rowlast );
#ifndef RELEASE
    type_ = typetmp;
#endif
}

template<typename Scalar>
inline bool
Dense<Scalar>::IsEmpty() const
{ return height_==0 || width_==0; }

template<typename Scalar>
inline void
Dense<Scalar>::Clear()
{
#ifndef RELEASE
    CallStackEntry entry("Dense::Clear");
#endif
#ifdef MEMORY_INFO
    AddToMemoryCount( -(double)memory_.size()*sizeof(Scalar) );
#endif
    height_ = 0;
    width_ = 0;
    ldim_ = 1;
    viewing_ = false;
    lockedView_ = false;


    std::vector<Scalar>().swap(memory_);
    buffer_ = 0;
    lockedBuffer_ = 0;
    type_ = GENERAL;
}

template<typename Scalar>
inline void
Dense<Scalar>::Set( int i, int j, Scalar value )
{
#ifndef RELEASE
    CallStackEntry entry("Dense::Set");
    if( lockedView_ )
        throw std::logic_error("Cannot change data in a locked view");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i >= height_ || j >= width_ )
        throw std::logic_error("Indices are out of bound");
    if( type_ == SYMMETRIC && j > i )
        throw std::logic_error("Setting upper entry from symmetric matrix");
#endif
    buffer_[i+j*ldim_] = value;
}

template<typename Scalar>
inline Scalar
Dense<Scalar>::Get( int i, int j ) const
{
#ifndef RELEASE
    CallStackEntry entry("Dense::Get");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i >= height_ || j >= width_ )
        throw std::logic_error("Indices are out of bound");
    if( type_ == SYMMETRIC && j > i )
        throw std::logic_error("Retrieving upper entry from symmetric matrix");
#endif
    if( lockedView_ )
        return lockedBuffer_[i+j*ldim_];
    else
        return buffer_[i+j*ldim_];
}

template<typename Scalar>
inline void
Dense<Scalar>::Print( const std::string tag, std::ostream& os ) const
{
#ifndef RELEASE
    CallStackEntry entry("Dense::Print");
#endif
    os.precision(15);
    os << tag << "\n";
    if( type_ == SYMMETRIC )
    {
        for( int i=0; i<height_; ++i )
        {
            for( int j=0; j<=i; ++j )
                os << WrapScalar(Get(i,j)) << " ";
            for( int j=i+1; j<width_; ++j )
                os << WrapScalar(Get(j,i)) << " ";
            os << "\n";
        }
    }
    else
    {
        for( int i=0; i<height_; ++i )
        {
            for( int j=0; j<width_; ++j )
                os << WrapScalar(Get(i,j)) << " ";
            os << "\n";
        }
    }
    os.flush();
}

template<typename Scalar>
inline void
Dense<Scalar>::Print2( const std::string tag, std::ostream& os ) const
{
#ifndef RELEASE
    CallStackEntry entry("Dense::Print");
#endif
    os.precision(5);
    os << tag << "\n";
    if( type_ == SYMMETRIC )
    {
        for( int i=0; i<height_; ++i )
        {
            for( int j=0; j<=i; ++j )
                os << Get(i,j) << " ";
            for( int j=i+1; j<width_; ++j )
                os << Get(j,i) << " ";
            os << "\n";
        }
    }
    else
    {
        for( int i=0; i<height_; ++i )
        {
            for( int j=0; j<width_; ++j )
                os << Get(i,j) << " ";
            os << "\n";
        }
    }
    os.flush();
}

template<typename Scalar>
inline Scalar*
Dense<Scalar>::Buffer( int i, int j )
{
#ifndef RELEASE
    CallStackEntry entry("Dense::Buffer");
    if( lockedView_ )
        throw std::logic_error("Cannot modify the buffer from a locked view");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > height_ || j > width_ )
        throw std::logic_error("Indices are out of bound");
#endif
    return &buffer_[i+j*ldim_];
}

template<typename Scalar>
inline const Scalar*
Dense<Scalar>::LockedBuffer( int i, int j ) const
{
#ifndef RELEASE
    CallStackEntry entry("Dense::LockedBuffer");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i > height_ || j > width_ )
        throw std::logic_error("Indices are out of bound");
#endif
    if( lockedView_ )
        return &lockedBuffer_[i+j*ldim_];
    else
        return &buffer_[i+j*ldim_];
}

template<typename Scalar>
inline void
Dense<Scalar>::View( Dense<Scalar>& A )
{
#ifndef RELEASE
    CallStackEntry entry("Dense::View");
#endif
    height_ = A.Height();
    width_ = A.Width();
    ldim_ = A.LDim();
    viewing_ = true;
    lockedView_ = false;
    buffer_ = A.Buffer();
    type_ = A.Type();
}

template<typename Scalar>
inline void
Dense<Scalar>::View( Dense<Scalar>& A, int i, int j, int height, int width )
{
#ifndef RELEASE
    CallStackEntry entry("Dense::View");
    if( A.Type() == SYMMETRIC && (i != j || height != width) )
        throw std::logic_error("Invalid submatrix of symmetric matrix");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i+height > A.Height() || j+width > A.Width() )
    {
        std::ostringstream s;
        s << "Submatrix out of bounds: attempted to grab ["
          << i << ":" << i+height-1 << "," << j << ":" << j+width-1
          << "] from " << A.Height() << " x " << A.Width() << " matrix.";
        throw std::logic_error( s.str().c_str() );
    }
#endif
    height_ = height;
    width_ = width;
    ldim_ = A.LDim();
    viewing_ = true;
    lockedView_ = false;
    buffer_ = A.Buffer(i,j);
    type_ = A.Type();
}

template<typename Scalar>
inline void
Dense<Scalar>::LockedView( const Dense<Scalar>& A )
{
#ifndef RELEASE
    CallStackEntry entry("Dense::LockedView");
#endif
    height_ = A.Height();
    width_ = A.Width();
    ldim_ = A.LDim();
    viewing_ = true;
    lockedView_ = true;
    lockedBuffer_ = A.LockedBuffer();
    type_ = A.Type();
}

template<typename Scalar>
inline void
Dense<Scalar>::LockedView
( const Dense<Scalar>& A, int i, int j, int height, int width )
{
#ifndef RELEASE
    CallStackEntry entry("Dense::LockedView");
    if( A.Type() == SYMMETRIC && (i != j || height != width) )
        throw std::logic_error("Invalid submatrix of symmetric matrix");
    if( i < 0 || j < 0 )
        throw std::logic_error("Indices must be non-negative");
    if( i+height > A.Height() || j+width > A.Width() )
    {
        std::ostringstream s;
        s << "Submatrix out of bounds: attempted to grab ["
          << i << ":" << i+height << "," << j << ":" << j+width
          << "] from " << A.Height() << " x " << A.Width() << " matrix.";
        throw std::logic_error( s.str().c_str() );
    }
#endif
    height_ = height;
    width_ = width;
    ldim_ = A.LDim();
    viewing_ = true;
    lockedView_ = true;
    lockedBuffer_ = A.LockedBuffer(i,j);
    type_ = A.Type();
}

template<typename Scalar>
inline void
Dense<Scalar>::Init()
{
#ifndef RELEASE
    CallStackEntry entry("Dense::Init");
    if( ldim_ < 0 || width_ < 0 )
        throw std::logic_error("Invalid dense matrix");
#endif
    std::memset( &memory_[0], 0, ldim_*width_*sizeof(Scalar) );
}

} // namespace hifde3d

#endif // ifndef HIFDE3D_DENSE_HPP

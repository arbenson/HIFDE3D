/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_VECTOR_HPP
#define DMHM_VECTOR_HPP 1

#include <cstring>
#include <stdexcept>
#include <vector>

namespace dmhm {

// A vector implementation that allows O(1) creation of subvectors.
template<typename T>
class Vector
{
    int height_;
    bool viewing_;
    bool lockedView_;
    std::vector<T> memory_;
    T* buffer_;
    const T* lockedBuffer_;

public:
    Vector();
    Vector( int height );
    Vector( int height, const T x );
    Vector( int height, T* buffer );
    Vector( int height, const T* lockedBuffer );
    Vector( const Vector<T>& x );
    ~Vector();

    int Height() const;
    int Size() const;
    void Resize( int height );
    void Clear();

    void Set( int i, T value );
    T Get( int i ) const;
    T & operator[]( int i );
    const T & operator[]( int i ) const;

    typedef typename std::vector<T>::iterator iterator;
    iterator Begin()
    { return memory_.begin(); }
    const iterator Begin() const
    { return memory_.begin(); }
    iterator End()
    { return memory_.end(); }
    const iterator End() const
    { return memory_.edu(); }

    void Erase( const iterator bp, const iterator ep );

    void PushBack( const T& x );
    void PushBack( T& x );
    void PopBack();

    void Print( const std::string tag, std::ostream& os=std::cout ) const;

    T* Buffer( int i=0 );
    const T* LockedBuffer( int i=0 ) const;

    void View( Vector<T>& x );
    void View( Vector<T>& x, int i, int height );

    void LockedView( const Vector<T>& x );
    void LockedView( const Vector<T>& x, int i, int height );
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

template<typename T>
inline
Vector<T>::Vector()
: height_(0), viewing_(false), lockedView_(false),
  memory_(), buffer_(0), lockedBuffer_(0)
{ }

template<typename T>
inline
Vector<T>::Vector( int height )
: height_(height), viewing_(false), lockedView_(false),
  memory_(height), buffer_(&memory_[0]), lockedBuffer_(0)
{
#ifdef MEMORY_INFO
    AddToMemoryCount( memory_.size()*sizeof(T) );
#endif
}

template<typename T>
inline
Vector<T>::Vector( int height, const T x )
: height_(height), viewing_(false), lockedView_(false),
  memory_(height), buffer_(&memory_[0]), lockedBuffer_(0)
{
    for( int i=0; i<height; ++i )
        memory_[i]=x;
#ifdef MEMORY_INFO
    AddToMemoryCount( memory_.size()*sizeof(T) );
#endif
}

template<typename T>
inline
Vector<T>::Vector( int height, T* buffer )
: height_(height), viewing_(true), lockedView_(false),
  memory_(), buffer_(buffer), lockedBuffer_(0)
{ }

template<typename T>
inline
Vector<T>::Vector( int height, const T* lockedBuffer )
: height_(height), viewing_(true), lockedView_(true),
  memory_(), buffer_(0), lockedBuffer_(lockedBuffer)
{ }

template<typename T>
inline
Vector<T>::Vector( const Vector<T>& x )
: height_(x.Height()), viewing_(false), lockedView_(false),
  memory_(x.Height()), buffer_(&memory_[0]), lockedBuffer_(0)
{ MemCopy( buffer_, x.LockedBuffer(), x.Height() ); }

template<typename T>
inline
Vector<T>::~Vector()
{
    Clear();
}

template<typename T>
inline int
Vector<T>::Height() const
{ return height_; }

template<typename T>
inline int
Vector<T>::Size() const
{ return height_; }

template<typename T>
inline void
Vector<T>::Resize( int height )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::Resize");
    if( viewing_ || lockedView_ )
        throw std::logic_error("Cannot resize a Vector that is a view.");
#endif
#ifdef MEMORY_INFO
    AddToMemoryCount( -(double)memory_.size()*sizeof(T) );
#endif
    height_ = height;
    memory_.resize( height );
    buffer_ = &memory_[0];
#ifdef MEMORY_INFO
    AddToMemoryCount( memory_.size()*sizeof(T) );
#endif
}

template<typename T>
inline void
Vector<T>::Clear()
{
#ifndef RELEASE
    CallStackEntry entry("Vector::Clear");
#endif
#ifdef MEMORY_INFO
    AddToMemoryCount( -(double)memory_.size()*sizeof(T) );
#endif
    height_ = 0;
    viewing_ = false;
    lockedView_ = false;

    std::vector<T>().swap(memory_);
    buffer_ = 0;
    lockedBuffer_ = 0;
}

template<typename T>
inline void
Vector<T>::Set( int i, T value )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::Set");
    if( lockedView_ )
        throw std::logic_error("Cannot modify locked views");
    if( i < 0 )
        throw std::logic_error("Negative buffer offsets are nonsensical");
    if( i >= height_ )
        throw std::logic_error("Vector::Set is out of bounds");
#endif
    buffer_[i] = value;
}

template<typename T>
inline T
Vector<T>::Get( int i ) const
{
#ifndef RELEASE
    CallStackEntry entry("Vector::Get");
    if( i < 0 )
        throw std::logic_error("Negative buffer offsets are nonsensical");
    if( i >= height_ )
        throw std::logic_error("Vector::Get is out of bounds");
#endif
    if( lockedView_ )
        return lockedBuffer_[i];
    else
        return buffer_[i];
}

template<typename T>
inline T &
Vector<T>::operator[]( int i )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::operator[]");
    if( i < 0 )
        throw std::logic_error("Negative buffer offsets are nonsensical");
    if( i > height_ && height_ > 0 )
        throw std::logic_error("Vector::operator[] is out of bounds");
    if( lockedView_ )
        throw std::logic_error("Vector::operator[] the memory is locked");
#endif
    return buffer_[i];
}

template<typename T>
inline const T &
Vector<T>::operator[]( int i ) const
{
#ifndef RELEASE
    CallStackEntry entry("Vector::operator[]");
    if( i < 0 )
        throw std::logic_error("Negative buffer offsets are nonsensical");
    if( i > height_ && height_ > 0 )
        throw std::logic_error("Vector::operator[] is out of bounds");
#endif
    if( lockedView_ )
        return lockedBuffer_[i];
    else
        return buffer_[i];
}

template<typename T>
inline void
Vector<T>::Erase( const iterator bp, const iterator ep )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::Erase");
#endif
#ifdef MEMORY_INFO
    AddToMemoryCount( -(double)memory_.size()*sizeof(T) );
#endif
    memory_.erase(bp,ep);
    height_ -= (ep-bp);
    buffer_=&memory_[0];
#ifdef MEMORY_INFO
    AddToMemoryCount( (double)memory_.size()*sizeof(T) );
#endif
}

template<typename T>
inline void
Vector<T>::PushBack( const T& x )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::PushBack");
#endif
#ifdef MEMORY_INFO
    AddToMemoryCount( -(double)memory_.size()*sizeof(T) );
#endif
    memory_.push_back(x);
    height_++;
    buffer_=&memory_[0];
#ifdef MEMORY_INFO
    AddToMemoryCount( (double)memory_.size()*sizeof(T) );
#endif
}

template<typename T>
inline void
Vector<T>::PushBack( T& x )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::PushBack");
#endif
#ifdef MEMORY_INFO
    AddToMemoryCount( -(double)memory_.size()*sizeof(T) );
#endif
    memory_.push_back(x);
    height_++;
    buffer_=&memory_[0];
#ifdef MEMORY_INFO
    AddToMemoryCount( (double)memory_.size()*sizeof(T) );
#endif
}

template<typename T>
inline void
Vector<T>::PopBack()
{
#ifndef RELEASE
    CallStackEntry entry("Vector::PopBack");
    if( height_ == 0 )
        throw std::logic_error("PopBack an empty Vector");
#endif
#ifdef MEMORY_INFO
    AddToMemoryCount( -(double)memory_.size()*sizeof(T) );
#endif
    memory_.pop_back();
    height_--;
    buffer_=&memory_[0];
#ifdef MEMORY_INFO
    AddToMemoryCount( (double)memory_.size()*sizeof(T) );
#endif
}

template<typename T>
inline void
Vector<T>::Print( const std::string tag, std::ostream& os ) const
{
#ifndef RELEASE
    CallStackEntry entry("Vector::Print");
#endif
    os << tag << "\n";
    if( lockedView_ )
    {
        for( int i=0; i<height_; ++i )
            os << WrapScalar(lockedBuffer_[i]) << "\n";
    }
    else
    {
        for( int i=0; i<height_; ++i )
            os << WrapScalar(buffer_[i]) << "\n";
    }
    os << std::endl;
}

template<typename T>
inline T*
Vector<T>::Buffer( int i )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::Buffer");
    if( lockedView_ )
        throw std::logic_error
                ("Cannot get modifiable buffer from locked view");
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
    if( i > height_ )
        throw std::logic_error("Out of bounds of buffer");
#endif
    return &buffer_[i];
}

template<typename T>
inline const T*
Vector<T>::LockedBuffer( int i ) const
{
#ifndef RELEASE
    CallStackEntry entry("Vector::LockedBuffer");
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
    if( i > height_ )
        throw std::logic_error("Out of bounds of buffer");
#endif
    if( lockedView_ )
        return &lockedBuffer_[i];
    else
        return &buffer_[i];
}

template<typename T>
inline void
Vector<T>::View( Vector<T>& x )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::View");
#endif
    viewing_ = true;
    lockedView_ = false;
    buffer_ = x.Buffer();
    height_ = x.Height();
}

template<typename T>
inline void
Vector<T>::View( Vector<T>& x, int i, int height )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::View");
    if( x.Height() < i+height )
        throw std::logic_error("Vector view goes out of bounds");
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
#endif
    viewing_ = true;
    lockedView_ = false;
    buffer_ = x.Buffer( i );
    height_ = height;
}

template<typename T>
inline void
Vector<T>::LockedView( const Vector<T>& x )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::LockedView");
#endif
    viewing_ = true;
    lockedView_ = true;
    lockedBuffer_ = x.Buffer();
    height_ = x.Height();
}

template<typename T>
inline void
Vector<T>::LockedView( const Vector<T>& x, int i, int height )
{
#ifndef RELEASE
    CallStackEntry entry("Vector::LockedView");
    if( x.Height() < i+height )
        throw std::logic_error("Vector view goes out of bounds");
    if( i < 0 )
        throw std::logic_error("Negative buffer offset is nonsensical");
#endif
    viewing_ = true;
    lockedView_ = true;
    lockedBuffer_ = x.LockedBuffer( i );
    height_ = height;
}

} // namespace dmhm

#endif // ifndef DMHM_VECTOR_HPP

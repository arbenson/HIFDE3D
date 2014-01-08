/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (HIFDE3D) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef HIFDE3D_MEMORY_MAP_HPP
#define HIFDE3D_MEMORY_MAP_HPP 1

namespace hifde3d {

template<typename T1,typename T2>
class MemoryMap
{
private:
    mutable unsigned currentIndex_;
    mutable typename std::map<T1,T2*>::iterator it_;
    mutable std::map<T1,T2*> baseMap_;
public:
    int Size() const { return baseMap_.size(); }
    void ResetIterator() { currentIndex_=0; it_=baseMap_.begin(); }
    int CurrentIndex() const { return currentIndex_; }

    T1 CurrentKey() const
    {
#ifndef RELEASE
        CallStackEntry entry("MemoryMap::CurrentKey");
        if( currentIndex_ >= baseMap_.size() )
            throw std::logic_error("Traversed past end of map");
#endif
        if( currentIndex_ == 0 )
            it_ = baseMap_.begin();
        return it_->first;
    }

    T2& Get( int key )
    {
#ifndef RELEASE
        CallStackEntry entry("MemoryMap::Get");
#endif
        T2* value = baseMap_[key];
#ifndef RELEASE
        if( value == 0 )
            throw std::logic_error("Tried to access with invalid key.");
#endif
        return *value;
    }

    const T2& Get( int key ) const
    {
#ifndef RELEASE
        CallStackEntry entry("MemoryMap::Get");
#endif
        T2* value = baseMap_[key];
#ifndef RELEASE
        if( value == 0 )
            throw std::logic_error("Tried to access with invalid key.");
#endif
        return *value;
    }


    void Set( int key, T2* value )
    {
#ifndef RELEASE
        CallStackEntry entry("MemoryMap::Set");
        if( baseMap_[key] != 0 )
            throw std::logic_error("Overwrote previous value");
#endif
        baseMap_[key] = value;
    }

    T2* CurrentEntry()
    {
#ifndef RELEASE
        CallStackEntry entry("MemoryMap::CurrentEntry");
        if( currentIndex_ >= baseMap_.size() )
            throw std::logic_error("Traversed past end of map");
#endif
        if( currentIndex_ == 0 )
            it_ = baseMap_.begin();

        T2* value = it_->second;
#ifndef RELEASE
        if( value == 0 )
            throw std::logic_error("Tried to return null pointer.");
#endif
        return value;
    }

    const T2* CurrentEntry() const
    {
#ifndef RELEASE
        CallStackEntry entry("MemoryMap::CurrentEntry");
        if( currentIndex_ >= baseMap_.size() )
            throw std::logic_error("Traversed past end of map");
#endif
        if( currentIndex_ == 0 )
            it_ = baseMap_.begin();

        const T2* value = it_->second;
#ifndef RELEASE
        if( value == 0 )
            throw std::logic_error("Tried to return null pointer.");
#endif
        return value;
    }
    //CurrentWidth and TotalWidth only supported when T2 has function Width
    const int CurrentWidth() const
    {
#ifndef RELEASE
        CallStackEntry entry("MemoryMap::CurrentWidth");
        if( currentIndex_ >= baseMap_.size() )
            throw std::logic_error("Traversed past end of map");
#endif
        if( currentIndex_ == 0 )
            it_ = baseMap_.begin();

        const T2* value = it_->second;
        return value->Width();
    }

    const int TotalWidth() const
    {
#ifndef RELEASE
        CallStackEntry entry("MemoryMap::TotalWidth");
#endif
        int width=0;
        typename std::map<T1,T2*>::iterator it = baseMap_.begin();
        for( unsigned int i=0; i<baseMap_.size(); ++i,++it)
            width += it->second->Width();
        return width;
    }

    const int EntrySize() const
    {
#ifndef RELEASE
        CallStackEntry entry("MemoryMap::EntrySize");
#endif
        int entSize=0;
        typename std::map<T1,T2*>::iterator it = baseMap_.begin();
        for( unsigned int i=0; i<baseMap_.size(); ++i,++it)
            entSize += it->second->Size();
        return entSize;
    }

    const int FirstWidth() const
    {
#ifndef RELEASE
        CallStackEntry entry("MemoryMap::FirstWidth");
#endif
        int width=0;
        if( baseMap_.size() > 0 )
        {
            typename std::map<T1,T2*>::iterator it = baseMap_.begin();
            width = it->second->Width();
        }
        return width;
    }

    void Increment()
    {
#ifndef RELEASE
        CallStackEntry entry("MemoryMap::Increment");
        if( currentIndex_ >= baseMap_.size() )
            throw std::logic_error("Traversed past end of map");
#endif
        if( currentIndex_ == 0 )
            it_ = baseMap_.begin();
        ++it_;
        ++currentIndex_;
    }

    void Decrement()
    {
#ifndef RELEASE
        CallStackEntry entry("MemoryMap::Decrement");
        if( currentIndex_ == 0 )
            throw std::logic_error("Traversed prior to beginning of map");
#endif
        --it_;
        --currentIndex_;
    }

    void Erase( int key )
    {
#ifndef RELEASE
        CallStackEntry entry("MemoryMap::Erase");
#endif
        delete baseMap_[key];
        baseMap_[key] = 0;
        baseMap_.erase( key );
        it_ = baseMap_.begin();
        currentIndex_ = 0;
    }

    void EraseCurrentEntry()
    {
#ifndef RELEASE
        CallStackEntry entry("MemoryMap::EraseCurrentEntry");
#endif
        delete it_->second;
        it_->second = 0;
        baseMap_.erase( it_++ );
    }

    void Clear()
    {
        typename std::map<T1,T2*>::iterator it;
        for( it=baseMap_.begin(); it!=baseMap_.end(); it++ )
        {
            delete it->second;
            it->second = 0;
        }
        baseMap_.clear();
    }

    MemoryMap() : currentIndex_(0) { }
    ~MemoryMap() { Clear(); }
};

} // namespace hifde3d

#endif // ifndef HIFDE3D_MEMORY_MAP_HPP

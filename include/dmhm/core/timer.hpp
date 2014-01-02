/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_TIMER_HPP
#define DMHM_TIMER_HPP 1

#include "mpi.hpp"

#include <map>

namespace dmhm {

// TODO: Switch to more reliable std::chrono-based C++11 implementation

class Timer
{
public:
    void Start( int key );
    double Stop( int key );
    double GetTime( int key );

    void Clear();
    void Clear( int key );
private:
    std::map<int,double> startTimes_, times_;
    std::map<int,bool> running_;
};

//----------------------------------------------------------------------------//
// Implementation begins here                                                 //
//----------------------------------------------------------------------------//

inline void
Timer::Start( int key )
{
#ifndef RELEASE
    CallStackEntry entry("Timer::Start");
#endif
    std::map<int,bool>::iterator it;
    it = running_.find( key );
    if( it == running_.end() )
    {
        running_[key] = true;
        startTimes_[key] = mpi::Time();
    }
    else
    {
        if( running_[key] )
            throw std::logic_error
            ("Restarted timer with same key without stopping");
        running_[key] = true;
        startTimes_[key] = mpi::Time();
    }
}

inline double
Timer::Stop( int key )
{
#ifndef RELEASE
    CallStackEntry entry("Timer::Stop");
#endif
    double pairTime = 0;

    std::map<int,bool>::iterator runningIt;
    runningIt = running_.find( key );
    if( runningIt == running_.end() || !running_[key] )
        throw std::logic_error("Stopped a timer that was not running");
    else
    {
        pairTime = mpi::Time() - startTimes_[key];

        std::map<int,double>::iterator timeIt;
        timeIt = times_.find( key );
        if( timeIt == times_.end() )
            times_[key] = pairTime;
        else
            times_[key] += pairTime;

        running_[key] = false;
    }
    return pairTime;
}

inline double
Timer::GetTime( int key )
{
#ifndef RELEASE
    CallStackEntry entry("Timer::GetTime");
#endif
    double time = 0;

    std::map<int,double>::iterator it;
    it = times_.find( key );
    if( it == times_.end() )
        time = 0;
    else
        time = times_[key];
    return time;
}

inline void Timer::Clear()
{
    running_.clear();
    startTimes_.clear();
    times_.clear();
}

inline void Timer::Clear( int key )
{
    running_.erase( key );
    startTimes_.erase( key );
    times_.erase( key );
}

} // namespace dmhm

#endif // ifndef DMHM_TIMER_HPP

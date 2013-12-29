/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (DMHM) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#pragma once
#ifndef DMHM_MPI_HPP
#define DMHM_MPI_HPP 1

#include "dmhm/config.h"
#include <complex>
#include <sstream>
#include <stdexcept>

#include "mpi.h"

namespace dmhm {
namespace mpi {

// Datatype definitions
typedef MPI_Comm Comm;
typedef MPI_Datatype Datatype;
typedef MPI_Errhandler ErrorHandler;
typedef MPI_Op Op;
typedef MPI_Request Request;
typedef MPI_Status Status;

// Standard constants
const int ANY_SOURCE = MPI_ANY_SOURCE;
const int ANY_TAG = MPI_ANY_TAG;
const int UNDEFINED = MPI_UNDEFINED;
const Comm COMM_SELF = MPI_COMM_SELF;
const Comm COMM_WORLD = MPI_COMM_WORLD;
const ErrorHandler ERRORS_RETURN = MPI_ERRORS_RETURN;
const ErrorHandler ERRORS_ARE_FATAL = MPI_ERRORS_ARE_FATAL;
const Request REQUEST_NULL = MPI_REQUEST_NULL;
const Op MAX = MPI_MAX;
const Op MIN = MPI_MIN;
const Op MAXLOC = MPI_MAXLOC;
const Op MINLOC = MPI_MINLOC;
const Op PROD = MPI_PROD;
const Op SUM = MPI_SUM;
const Op LOGICAL_AND = MPI_LAND;
const Op LOGICAL_OR = MPI_LOR;
const Op LOGICAL_XOR = MPI_LXOR;
const Op BINARY_AND = MPI_BAND;
const Op BINARY_OR = MPI_BOR;
const Op BINARY_XOR = MPI_BXOR;

// Added constant(s)
const int MIN_COLL_MSG = 1; // minimum message size for collectives

//----------------------------------------------------------------------------//
// Routines                                                                   //
//----------------------------------------------------------------------------//

// Environment routines
void Initialize( int& argc, char**& argv );
void Finalize();
bool Initialized();
bool Finalized();
double Time();

// Communicator manipulation
int CommRank( Comm comm );
int CommSize( Comm comm );
void CommDup( Comm original, Comm& duplicate );
void CommSplit( Comm comm, int color, int key, Comm& newComm );
void CommFree( Comm& comm );
void ErrorHandlerSet( Comm comm, ErrorHandler errorHandler );

// Point-to-point communication
template<typename R>
void Send( const R* buf, int count, int to, int tag, Comm comm );
template<typename R>
void Send( const std::complex<R>* buf, int count, int to, int tag, Comm comm );

template<typename R>
void ISend
( const R* buf, int count, int to, int tag, Comm comm, Request& request );
template<typename R>
void ISend
( const std::complex<R>* buf, int count, int to, int tag, Comm comm,
  Request& request );

template<typename R>
void Recv( R* buf, int count, int from, int tag, Comm comm );
template<typename R>
void Recv( std::complex<R>* buf, int count, int from, int tag, Comm comm );

template<typename R>
void IRecv
( R* buf, int count, int from, int tag, Comm comm, Request& request );
template<typename R>
void IRecv
( std::complex<R>* buf, int count, int from, int tag, Comm comm, Request& request );

template<typename R>
void SendRecv
( const R* sbuf, int sc, int to,   int stag,
        R* rbuf, int rc, int from, int rtag, Comm comm );
template<typename R>
void SendRecv
( const std::complex<R>* sbuf, int sc, int to,   int stag,
        std::complex<R>* rbuf, int rc, int from, int rtag, Comm comm );

void Wait( MPI_Request& request );
void Wait( Request& request, Status& status );
void WaitAll( int numRequests, Request* requests );
void WaitAll( int numRequests, Request* requests, Status* statuses );

// Collective communication

void Barrier( Comm comm );

template<typename R>
void Broadcast( R* buf, int count, int root, Comm comm );
template<typename R>
void Broadcast( std::complex<R>* buf, int count, int root, Comm comm );

template<typename R>
void AllGather
( const R* sbuf, int sc,
        R* rbuf, int rc, Comm comm );
template<typename R>
void AllGather
( const std::complex<R>* sbuf, int sc,
        std::complex<R>* rbuf, int rc, Comm comm );

template<typename T>
void Reduce
( const T* sbuf, T* rbuf, int count, Op op, int root, Comm comm );
template<typename R>
void Reduce
( const std::complex<R>* sbuf, std::complex<R>* rbuf, int count, Op op,
  int root, Comm comm );

// In-place option
template<typename T>
void Reduce( T* buf, int count, Op op, int root, Comm comm );
template<typename R>
void Reduce( std::complex<R>* buf, int count, Op op, int root, Comm comm );

template<typename T>
void AllReduce( const T* sbuf, T* rbuf, int count, Op op, Comm comm );
template<typename R>
void AllReduce
( const std::complex<R>* sbuf, std::complex<R>* rbuf, int count, Op op, Comm comm );

template<typename R>
void AllToAll
( const R* sbuf, int sc,
        R* rbuf, int rc, Comm comm );
template<typename R>
void AllToAll
( const std::complex<R>* sbuf, int sc,
        std::complex<R>* rbuf, int rc, Comm comm );

} // namespace mpi
} // namespace dmhm

#endif // ifndef DMHM_MPI_HPP

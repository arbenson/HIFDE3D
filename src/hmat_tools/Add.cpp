// Dense C := alpha A + beta B
template<typename Scalar>
void Add
( Scalar alpha, const Dense<Scalar>& A,
  Scalar beta,  const Dense<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Add (D := D + D)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Tried to add nonconforming matrices.");
    // TODO: Allow for A and B to have different types
    if( A.Type() != B.Type() )
        throw std::logic_error("Add with different types not written");
#endif
    const int m = A.Height();
    const int n = A.Width();

    C.SetType( A.Type() );
    C.Resize( m, n );

    if( C.Symmetric() )
    {
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT ACol = A.LockedBuffer(0,j);
            const Scalar* RESTRICT BCol = B.LockedBuffer(0,j);
            Scalar* RESTRICT CCol = C.Buffer(0,j);
            for( int i=j; i<m; ++i )
                CCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
    else
    {
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT ACol = A.LockedBuffer(0,j);
            const Scalar* RESTRICT BCol = B.LockedBuffer(0,j);
            Scalar* RESTRICT CCol = C.Buffer(0,j);
            for( int i=0; i<m; ++i )
                CCol[i] = alpha*ACol[i] + beta*BCol[i];
        }
    }
}

// Dense C := alpha A + beta B
template<typename Scalar>
void Axpy
( Scalar alpha, const Dense<Scalar>& A,
                Dense<Scalar>& B )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Axpy (D := D + D)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Tried to add nonconforming matrices.");
#endif
    const int m = A.Height();
    const int n = A.Width();

    for( int j=0; j<n; ++j )
    {
        const Scalar* RESTRICT ACol = A.LockedBuffer(0,j);
        Scalar* RESTRICT BCol = B.Buffer(0,j);
        for( int i=0; i<m; ++i )
            BCol[i] = alpha*ACol[i] + BCol[i];
    }
}

// Low-rank C := alpha A + beta B
template<typename Scalar>
void Add
( Scalar alpha, const LowRank<Scalar>& A,
  Scalar beta,  const LowRank<Scalar>& B,
                      LowRank<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Add (F := F + F)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Tried to add nonconforming matrices.");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int Ar = A.Rank();
    const int Br = B.Rank();
    const int r = Ar + Br;
    C.U.SetType( GENERAL ); C.U.Resize( m, r );
    C.V.SetType( GENERAL ); C.V.Resize( n, r );

    // C.U := [(alpha A.U), (beta B.U)]
    // Copy in (alpha A.U)
    for( int j=0; j<Ar; ++j )
    {
        Scalar* RESTRICT CUACol = C.U.Buffer(0,j);
        const Scalar* RESTRICT AUCol = A.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            CUACol[i] = alpha*AUCol[i];
    }
    // Copy in (beta B.U)
    for( int j=0; j<Br; ++j )
    {
        Scalar* RESTRICT CUBCol = C.U.Buffer(0,j+Ar);
        const Scalar* RESTRICT BUCol = B.U.LockedBuffer(0,j);
        for( int i=0; i<m; ++i )
            CUBCol[i] = beta*BUCol[i];
    }

    // C.V := [A.V B.V]
    for( int j=0; j<Ar; ++j )
        MemCopy( C.V.Buffer(0,j), A.V.LockedBuffer(0,j), n );

    for( int j=0; j<Br; ++j )
        MemCopy( C.V.Buffer(0,j+Ar), B.V.LockedBuffer(0,j), n );
}

// Dense from sum of low-rank and dense:  C := alpha A + beta B
template<typename Scalar>
void Add
( Scalar alpha, const LowRank<Scalar>& A,
  Scalar beta,  const Dense<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Add (D := F + D)");
    if( A.Height() != B.Height() || A.Width() != B.Width()  )
        throw std::logic_error("Tried to add nonconforming matrices.");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int r = A.Rank();

    C.SetType( GENERAL );
    C.Resize( m, n );

    if( B.Symmetric() )
    {
        // Form the full C := beta B from the symmetric B

        // Form the lower triangle
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT BCol = B.LockedBuffer(0,j);
            Scalar* RESTRICT CCol = C.Buffer(0,j);
            for( int i=j; i<n; ++i )
                CCol[i] = beta*BCol[i];
        }

        // Transpose the strictly-lower triangle into the upper triangle
        const int ldc = C.LDim();
        for( int j=0; j<n-1; ++j )
        {
            const Scalar* CCol = C.LockedBuffer(0,j);
            Scalar* CRow = C.Buffer(j,0);
            for( int i=j+1; i<n; ++i )
                CRow[i*ldc] = CCol[i];
        }

        // C := alpha A + C = alpha A.U A.V^[T,H] + C
        const char option = 'T';
        blas::Gemm
        ( 'N', option, m, n, r,
          alpha, A.U.LockedBuffer(), A.U.LDim(),
                 A.V.LockedBuffer(), A.V.LDim(),
          1,     C.Buffer(),         C.LDim() );
    }
    else
    {
        // Form C := beta B
        for( int j=0; j<n; ++j )
        {
            const Scalar* RESTRICT BCol = B.LockedBuffer(0,j);
            Scalar* RESTRICT CCol = C.Buffer(0,j);
            for( int i=0; i<m; ++i )
                CCol[i] = beta*BCol[i];
        }

        // C := alpha A + C = alpha A.U A.V^[T,H] + C
        const char option = 'T';
        blas::Gemm
        ( 'N', option, m, n, r,
          alpha, A.U.LockedBuffer(), A.U.LDim(),
                 A.V.LockedBuffer(), A.V.LDim(),
          1,     C.Buffer(),         C.LDim() );
    }
}

// Dense from sum of dense and low-rank:  C := alpha A + beta B
// The arguments are switched for generality, so just call the other version.
template<typename Scalar>
void Add
( Scalar alpha, const Dense<Scalar>& A,
  Scalar beta,  const LowRank<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Add (D := D + F)");
#endif
    Add( beta, B, alpha, A, C );
}

// Dense as sum of two low-rank matrices
template<typename Scalar>
void Add
( Scalar alpha, const LowRank<Scalar>& A,
  Scalar beta,  const LowRank<Scalar>& B,
                      Dense<Scalar>& C )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::Add (D := F + F)");
    if( A.Height() != B.Height() || A.Width() != B.Width() )
        throw std::logic_error("Tried to add nonconforming matrices.");
#endif
    const int m = A.Height();
    const int n = A.Width();
    const int r = A.Rank();

    C.SetType( GENERAL );
    C.Resize( m, n );

    // C := alpha A = alpha A.U A.V^[T,H] + C
    const char option = 'T';
    blas::Gemm
    ( 'N', option, m, n, r,
      alpha, A.U.LockedBuffer(), A.U.LDim(),
             A.V.LockedBuffer(), A.V.LDim(),
      0,     C.Buffer(),         C.LDim() );
    // C := beta B + C = beta B.U B.V^[T,H] + C
    blas::Gemm
    ( 'N', option, m, n, r,
      beta, B.U.LockedBuffer(), B.U.LDim(),
            B.V.LockedBuffer(), B.V.LDim(),
      1,    C.Buffer(),         C.LDim() );
}

// Dense C := alpha A + beta B
template void Add
( float alpha, const Dense<float>& A,
  float beta,  const Dense<float>& B,
                     Dense<float>& C );
template void Add
( double alpha, const Dense<double>& A,
  double beta,  const Dense<double>& B,
                      Dense<double>& C );
template void Add
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
  std::complex<float> beta,  const Dense<std::complex<float> >& B,
                                   Dense<std::complex<float> >& C );
template void Add
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
  std::complex<double> beta,  const Dense<std::complex<double> >& B,
                                    Dense<std::complex<double> >& C );

// Dense C := alpha A + beta B
template void Axpy
( float alpha, const Dense<float>& A,
               Dense<float>& B );
template void Axpy
( double alpha, const Dense<double>& A,
                Dense<double>& B );
template void Axpy
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
                             Dense<std::complex<float> >& B );
template void Axpy
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
                              Dense<std::complex<double> >& B );

// Low-rank C := alpha A + beta B
template void Add
( float alpha, const LowRank<float>& A,
  float beta,  const LowRank<float>& B,
                     LowRank<float>& C );
template void Add
( double alpha, const LowRank<double>& A,
  double beta,  const LowRank<double>& B,
                      LowRank<double>& C );
template void Add
( std::complex<float> alpha, const LowRank<std::complex<float> >& A,
  std::complex<float> beta,  const LowRank<std::complex<float> >& B,
                                   LowRank<std::complex<float> >& C );
template void Add
( std::complex<double> alpha, const LowRank<std::complex<double> >& A,
  std::complex<double> beta,  const LowRank<std::complex<double> >& B,
                                    LowRank<std::complex<double> >& C );

// Dense as sum of low-rank and dense, C := alpha A + beta B
template void Add
( float alpha, const LowRank<float>& A,
  float beta,  const Dense<float>& B,
                     Dense<float>& C );
template void Add
( double alpha, const LowRank<double>& A,
  double beta,  const Dense<double>& B,
                      Dense<double>& C );
template void Add
( std::complex<float> alpha, const LowRank<std::complex<float> >& A,
  std::complex<float> beta,  const Dense<std::complex<float> >& B,
                                   Dense<std::complex<float> >& C );
template void Add
( std::complex<double> alpha, const LowRank<std::complex<double> >& A,
  std::complex<double> beta,  const Dense<std::complex<double> >& B,
                                    Dense<std::complex<double> >& C );

// Dense as sum of dense and low-rank, C := alpha A + beta B
template void Add
( float alpha, const Dense<float>& A,
  float beta,  const LowRank<float>& B,
                     Dense<float>& C );
template void Add
( double alpha, const Dense<double>& A,
  double beta,  const LowRank<double>& B,
                      Dense<double>& C );
template void Add
( std::complex<float> alpha, const Dense<std::complex<float> >& A,
  std::complex<float> beta,  const LowRank<std::complex<float> >& B,
                                   Dense<std::complex<float> >& C );
template void Add
( std::complex<double> alpha, const Dense<std::complex<double> >& A,
  std::complex<double> beta,  const LowRank<std::complex<double> >& B,
                                    Dense<std::complex<double> >& C );

// Dense as sum of two low-rank matrices
template void Add
( float alpha, const LowRank<float>& A,
  float beta,  const LowRank<float>& B,
                     Dense<float>& C );
template void Add
( double alpha, const LowRank<double>& A,
  double beta,  const LowRank<double>& B,
                      Dense<double>& C );
template void Add
( std::complex<float> alpha, const LowRank<std::complex<float> >& A,
  std::complex<float> beta,  const LowRank<std::complex<float> >& B,
                                   Dense<std::complex<float> >& C );
template void Add
( std::complex<double> alpha, const LowRank<std::complex<double> >& A,
  std::complex<double> beta,  const LowRank<std::complex<double> >& B,
                                    Dense<std::complex<double> >& C );


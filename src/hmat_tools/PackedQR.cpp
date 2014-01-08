/*
   Copyright (c) 2011-2013 Jack Poulson, Lexing Ying,
   The University of Texas at Austin, and Stanford University

   This file is part of Distributed-Memory Hierarchical Matrices (HIFDE3D) and is
   under the GPLv3 License, which can be found in the LICENSE file in the root
   directory, or at http://opensource.org/licenses/GPL-3.0
*/
#include "hifde3d.hpp"

namespace {

// Compute a Householder vector in-place and return the tau factor

template<typename Real>
Real Householder( const int m, Real* buffer )
{
    if( m == 1 )
    {
        buffer[0] = -buffer[0];
        return (Real)2;
    }

    Real alpha = buffer[0];
    Real norm = hifde3d::blas::Nrm2( m-1, &buffer[1], 1 );

    Real beta;
    if( alpha <= 0 )
        beta = hifde3d::lapack::SafeNorm( alpha, norm );
    else
        beta = -hifde3d::lapack::SafeNorm( alpha, norm );

    // Avoid overflow by scaling the vector
    const Real safeMin = hifde3d::lapack::MachineSafeMin<Real>() /
                         hifde3d::lapack::MachineEpsilon<Real>();
    int count = 0;
    if( hifde3d::Abs(beta) < safeMin )
    {
        Real invOfSafeMin = static_cast<Real>(1) / safeMin;
        do
        {
            ++count;
            hifde3d::blas::Scal( m-1, invOfSafeMin, &buffer[1], 1 );
            alpha *= invOfSafeMin;
            beta *= invOfSafeMin;
        } while( hifde3d::Abs( beta ) < safeMin );

        norm = hifde3d::blas::Nrm2( m-1, &buffer[1], 1 );
        if( alpha <= 0 )
            beta = hifde3d::lapack::SafeNorm( alpha, norm );
        else
            beta = -hifde3d::lapack::SafeNorm( alpha, norm );
    }

    Real tau = ( beta - alpha ) / beta;
    hifde3d::blas::Scal( m-1, static_cast<Real>(1)/(alpha-beta), &buffer[1], 1 );

    // Rescale the vector
    for( int j=0; j<count; ++j )
        beta *= safeMin;
    buffer[0] = beta;

    return tau;
}

template<typename Real>
std::complex<Real> Householder( const int m, std::complex<Real>* buffer )
{
    typedef std::complex<Real> Scalar;

    Scalar alpha = buffer[0];
    Real norm = hifde3d::blas::Nrm2( m-1, &buffer[1], 1 );

    if( norm == 0 && imag(alpha) == (Real)0 )
    {
        buffer[0] = -buffer[0];
        return (Real)2;
    }

    Real beta;
    if( real(alpha) <= 0 )
        beta = hifde3d::lapack::SafeNorm( real(alpha), imag(alpha), norm );
    else
        beta = -hifde3d::lapack::SafeNorm( real(alpha), imag(alpha), norm );

    // Avoid overflow by scaling the vector
    const Real safeMin = hifde3d::lapack::MachineSafeMin<Real>() /
                         hifde3d::lapack::MachineEpsilon<Real>();
    int count = 0;
    if( hifde3d::Abs(beta) < safeMin )
    {
        Real invOfSafeMin = static_cast<Real>(1) / safeMin;
        do
        {
            ++count;
            hifde3d::blas::Scal( m-1, Scalar(invOfSafeMin), &buffer[1], 1 );
            alpha *= invOfSafeMin;
            beta *= invOfSafeMin;
        } while( hifde3d::Abs( beta ) < safeMin );

        norm = hifde3d::blas::Nrm2( m-1, &buffer[1], 1 );
        if( real(alpha) <= 0 )
            beta = hifde3d::lapack::SafeNorm( real(alpha), imag(alpha), norm );
        else
            beta = -hifde3d::lapack::SafeNorm( real(alpha), imag(alpha), norm );
    }

    Scalar tau = Scalar( (beta-real(alpha))/beta, -imag(alpha)/beta );
    hifde3d::blas::Scal( m-1, static_cast<Scalar>(1)/(alpha-beta), &buffer[1], 1 );

    // Rescale the vector
    for( int j=0; j<count; ++j )
        beta *= safeMin;
    buffer[0] = beta;

    return tau;
}

} // anonymous namespace

namespace hifde3d {
namespace hmat_tools {

// Perform a QR factorization of size (s+t) x r where only the upper triangles
// of the s x r and t x r submatrices are nonzero, and the nonzeros are packed
// columnwise.
//
// The work buffer must be of size t-1.
template<typename Scalar>
void PackedQR
( const int r, const int s, const int t,
  Scalar* RESTRICT packedA, Scalar* RESTRICT tau, Scalar* RESTRICT work )
{
    const int minDim = std::min(s+t,r);

    int jCol = 0;
    for( int j=0; j<minDim; ++j )
    {
        const int S = std::min(j+1,s);
        const int T = std::min(j+1,t);
        const int overlap = ( j >= s ? j+1-s : 0 );

        // Compute the Householder vector, v, and scalar, tau, in-place
        const int jDiag = jCol + j;
        tau[j] = Householder( S+T-j, &packedA[jDiag] );

        // Form z := A(I_j,j+1:end)' v in the work vector
        int iCol = jCol + S + T;
        for( int i=0; i<r-(j+1); ++i )
        {
            const int Si = std::min(j+i+2,s);
            const int Ti = std::min(j+i+2,t);

            // z[i] := Conj(A(j,j+i+1)) v(0) = Conj(A(j,j+i+1))
            const int iDiagRight = iCol + j;
            work[i] = Conj(packedA[iDiagRight]);

            // Traverse over this col of the lower triangle
            const int jump = ( j >= s ? 1 : Si-j );
            for( int k=0; k<T-overlap; ++k )
                work[i] += Conj(packedA[iDiagRight+k+jump])*packedA[jDiag+k+1];

            iCol += Si + Ti;
        }

        // A(I_j,j+1:end) -= conj(tau) v z'
        iCol = jCol + S + T;
        for( int i=0; i<r-(j+1); ++i )
        {
            const int Si = std::min(j+i+2,s);
            const int Ti = std::min(j+i+2,t);

            const Scalar scale = Conj(tau[j])*Conj(work[i]);

            // A(j,j+i+1) -= conj(tau) v(0) z[k] = conj(tau) z[k]
            const int iDiagRight = iCol + j;
            packedA[iDiagRight] -= scale;

            // Traverse over the relevant piece of this col of the
            // lower-triangle
            const int jump = ( j >= s ? 1 : Si-j );
            for( int k=0; k<T-overlap; ++k )
                packedA[iDiagRight+k+jump] -= scale*packedA[jDiag+k+1];

            iCol += Si + Ti;
        }

        jCol += S + T;
    }
}

template<typename Scalar>
void ApplyPackedQFromLeft
( const int r, const int s, const int t,
  const Scalar* RESTRICT packedA, const Scalar* RESTRICT tau,
  Dense<Scalar>& B, Scalar* work )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::ApplyPackedQFromLeft");
    if( B.Type() != GENERAL )
        throw std::logic_error("B must be a full dense matrix");
    if( B.Height() != s+t )
        throw std::logic_error("B is not the correct height");
#endif
    Scalar* BBuffer = B.Buffer();
    const int n = B.Width();
    const int BLDim = B.LDim();
    const int minDim = std::min(s+t,r);

    int jCol = (s*s+s)/2 + (minDim-s)*s + (t*t+t)/2 + (minDim-t)*t;
    for( int j=minDim-1; j>=0; --j )
    {
        const int S = std::min(j+1,s);
        const int T = std::min(j+1,t);
        const int overlap = ( j >= s ? j+1-s : 0 );
        jCol -= S + T;

        // B := (I - tau_j v_j v_j') B
        //    = B - tau_j v_j (v_j' B)
        //    = B - tau_j v_j (B' v_j)'

        // 1) Form w_j := B' v_j
        // Since v_j's only nonzero entries are a 1 in the j'th entry and
        // arbitrary values in the r:r+j entries,
        //     w_j = B(j,:)' + B(s:s+T-1,:)' v_j(s:s+T-1)
        for( int i=0; i<n; ++i )
            work[i] = Conj(BBuffer[j+i*BLDim]);
        blas::Gemv
        ( 'C', T-overlap, n,
          Scalar(1), &BBuffer[s+overlap],      BLDim,
                     &packedA[jCol+S+overlap], 1,
          Scalar(1), work,                     1 );

        // 2) B := B - tau_j v_j w_j'
        // Since v_j has the structure described above, we only need to
        // subtract tau_j w_j' from the j'th row of B and then perform the
        // update
        //     B(s:s+T-1,:) -= tau_j v_j(s:s+T-1) w_j'
        const Scalar tauj = tau[j];
        for( int i=0; i<n; ++i )
            BBuffer[j+i*BLDim] -= tauj*Conj(work[i]);
        blas::Ger
        ( T-overlap, n,
          -tauj, &packedA[jCol+S+overlap], 1,
                 work,                     1,
                 &BBuffer[s+overlap],      BLDim );
    }
}

template<typename Scalar>
void ApplyPackedQAdjointFromLeft
( const int r, const int s, const int t,
  const Scalar* RESTRICT packedA, const Scalar* RESTRICT tau,
  Dense<Scalar>& B, Scalar* work )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::ApplyPackedQAdjointFromLeft");
    if( B.Type() != GENERAL )
        throw std::logic_error("B must be a full dense matrix");
    if( B.Height() != s+t )
        throw std::logic_error("B is not the correct height");
#endif
    Scalar* BBuffer = B.Buffer();
    const int n = B.Width();
    const int BLDim = B.LDim();
    const int minDim = std::min(s+t,r);

    int jCol = 0;
    for( int j=0; j<minDim; ++j )
    {
        const int S = std::min(j+1,s);
        const int T = std::min(j+1,t);
        const int overlap = ( j >= s ? j+1-s : 0 );

        // B := (I - conj(tau_j) v_j v_j') B
        //    = B - conj(tau_j) v_j v_j' B
        //    = B - conj(tau_j) v_j (B' v_j)'

        // 1) Form w_j := B' v_j
        // Since v_j's only nonzero entries are a 1 in the j'th entry and
        // arbitrary values in the r:r+j entries,
        //     w_j = B(j,:)' + B(s:s+T-1,:)' v_j(s:s+T-1)
        for( int i=0; i<n; ++i )
            work[i] = Conj(BBuffer[j+i*BLDim]);
        blas::Gemv
        ( 'C', T-overlap, n,
          Scalar(1), &BBuffer[s+overlap],      BLDim,
                     &packedA[jCol+S+overlap], 1,
          Scalar(1), work,                     1 );

        // 2) B := B - tau_j v_j w_j'
        // Since v_j has the structure described above, we only need to
        // subtract tau_j w_j' from the j'th row of B and then perform the
        // update
        //     B(s:s+T-1,:) -= tau_j v_j(s:s+T-1) w_j'
        const Scalar conjTauj = Conj(tau[j]);
        for( int i=0; i<n; ++i )
            BBuffer[j+i*BLDim] -= conjTauj*Conj(work[i]);
        blas::Ger
        ( T-overlap, n,
          -conjTauj, &packedA[jCol+S+overlap], 1,
                     work,                     1,
                     &BBuffer[s+overlap],      BLDim );

        jCol += S + T;
    }
}

template<typename Scalar>
void ApplyPackedQFromRight
( const int r, const int s, const int t,
  const Scalar* RESTRICT packedA, const Scalar* RESTRICT tau,
  Dense<Scalar>& B, Scalar* work )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::ApplyPackedQFromRight");
    if( B.Type() != GENERAL )
        throw std::logic_error("B must be a full dense matrix");
    if( B.Width() != s+t )
        throw std::logic_error("B is not the correct width");
#endif
    Scalar* BBuffer = B.Buffer();
    const int m = B.Height();
    const int BLDim = B.LDim();
    const int minDim = std::min(s+t,r);

    int jCol = 0;
    for( int j=0; j<minDim; ++j )
    {
        const int S = std::min(j+1,s);
        const int T = std::min(j+1,t);
        const int overlap = ( j >= s ? j+1-s : 0 );

        // B := B (I - tau_j v_j v_j')
        //    = B - (tau_j B v_j) v_j'

        // 1) Form w_j := tau_j B v_j
        const Scalar tauj = tau[j];
        for( int i=0; i<m; ++i )
            work[i] = tauj*BBuffer[i+j*BLDim];
        blas::Gemv
        ( 'N', m, T-overlap,
          tauj,      &BBuffer[(s+overlap)*BLDim], BLDim,
                     &packedA[jCol+S+overlap],    1,
          Scalar(1), work,                        1 );

        // 2) B := B - w_j v_j'
        for( int i=0; i<m; ++i )
            BBuffer[i+j*BLDim] -= work[i];
        blas::Ger
        ( m, T-overlap,
         Scalar(-1), work,                        1,
                     &packedA[jCol+S+overlap],    1,
                     &BBuffer[(s+overlap)*BLDim], BLDim );

        jCol += S + T;
    }
}

template<typename Scalar>
void ApplyPackedQAdjointFromRight
( const int r, const int s, const int t,
  const Scalar* RESTRICT packedA, const Scalar* RESTRICT tau,
  Dense<Scalar>& B, Scalar* work )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::ApplyPackedQAdjointFromRight");
    if( B.Type() != GENERAL )
        throw std::logic_error("B must be a full dense matrix");
    if( B.Width() != s+t )
        throw std::logic_error("B is not the correct width");
#endif
    Scalar* BBuffer = B.Buffer();
    const int m = B.Height();
    const int BLDim = B.LDim();
    const int minDim = std::min(s+t,r);

    int jCol = (s*s+s)/2 + (minDim-s)*s + (t*t+t)/2 + (minDim-t)*t;
    for( int j=minDim-1; j>=0; --j )
    {
        const int S = std::min(j+1,s);
        const int T = std::min(j+1,t);
        const int overlap = ( j >= s ? j+1-s : 0 );
        jCol -= S + T;

        // B := B (I - conj(tau)_j v_j v_j')
        //    = B - (conj(tau_j) B v_j) v_j'

        // 1) Form w_j := conj(tau_j) B v_j
        const Scalar conjTauj = Conj(tau[j]);
        for( int i=0; i<m; ++i )
            work[i] = conjTauj*BBuffer[i+j*BLDim];
        blas::Gemv
        ( 'N', m, T-overlap,
          conjTauj,  &BBuffer[(s+overlap)*BLDim], BLDim,
                     &packedA[jCol+S+overlap],    1,
          Scalar(1), work,              1 );

        // 2) B := B - w_j v_j'
        for( int i=0; i<m; ++i )
            BBuffer[i+j*BLDim] -= work[i];
        blas::Ger
        ( m, T-overlap,
         Scalar(-1), work,                        1,
                     &packedA[jCol+S+overlap],    1,
                     &BBuffer[(s+overlap)*BLDim], BLDim );
    }
}

template<typename Scalar>
void PrintPacked
( const std::string msg,
  const int r, const int s, const int t,
  const Scalar* packedA, std::ostream& os )
{
#ifndef RELEASE
    CallStackEntry entry("hmat_tools::PrintPacked");
#endif
    os << msg << "\n";

    // Print the upper triangle
    int iCol = 0;
    for( int i=0; i<s; ++i )
    {
        const int Si = std::min(i+1,s);
        const int Ti = std::min(i+1,t);

        for( int j=0; j<i; ++j )
            os << "0 ";

        int jCol = iCol;
        for( int j=i; j<r; ++j )
        {
            const int Sj = std::min(j+1,s);
            const int Tj = std::min(j+1,t);

            const Scalar value = packedA[jCol+i];
            os << WrapScalar(value) << " ";

            jCol += Sj + Tj;
        }
        os << "\n";

        iCol += Si + Ti;
    }

    // Print the lower triangle
    iCol = 0;
    for( int i=0; i<t; ++i )
    {
        const int Si = std::min(i+1,s);
        const int Ti = std::min(i+1,t);

        for( int j=0; j<i; ++j )
            os << "0 ";

        int jCol = iCol;
        for( int j=i; j<r; ++j )
        {
            const int Sj = std::min(j+1,s);
            const int Tj = std::min(j+1,t);

            const Scalar value = packedA[jCol+Sj+i];
            os << WrapScalar(value) << " ";

            jCol += Sj + Tj;
        }
        os << "\n";

        iCol += Si + Ti;
    }

    os.flush();
}

template void PackedQR
( const int r, const int s, const int t,
  float* RESTRICT A,
  float* RESTRICT tau,
  float* RESTRICT work );
template void PackedQR
( const int r, const int s, const int t,
  double* RESTRICT A,
  double* RESTRICT tau,
  double* RESTRICT work );
template void PackedQR
( const int r, const int s, const int t,
  std::complex<float>* RESTRICT A,
  std::complex<float>* RESTRICT tau,
  std::complex<float>* RESTRICT work );
template void PackedQR
( const int r, const int s, const int t,
  std::complex<double>* RESTRICT A,
  std::complex<double>* RESTRICT tau,
  std::complex<double>* RESTRICT work );

template void ApplyPackedQFromLeft
( const int r, const int s, const int t,
  const float* RESTRICT A,
  const float* RESTRICT tau,
        Dense<float>& B,
        float* RESTRICT work );
template void ApplyPackedQFromLeft
( const int r, const int s, const int t,
  const double* RESTRICT A,
  const double* RESTRICT tau,
        Dense<double>& B,
        double* RESTRICT work );
template void ApplyPackedQFromLeft
( const int r, const int s, const int t,
  const std::complex<float>* RESTRICT A,
  const std::complex<float>* RESTRICT tau,
        Dense<std::complex<float> >& B,
        std::complex<float>* RESTRICT work );
template void ApplyPackedQFromLeft
( const int r, const int s, const int t,
  const std::complex<double>* RESTRICT A,
  const std::complex<double>* RESTRICT tau,
        Dense<std::complex<double> >& B,
        std::complex<double>* RESTRICT work );

template void ApplyPackedQAdjointFromLeft
( const int r, const int s, const int t,
  const float* RESTRICT A,
  const float* RESTRICT tau,
        Dense<float>& B,
        float* RESTRICT work );
template void ApplyPackedQAdjointFromLeft
( const int r, const int s, const int t,
  const double* RESTRICT A,
  const double* RESTRICT tau,
        Dense<double>& B,
        double* RESTRICT work );
template void ApplyPackedQAdjointFromLeft
( const int r, const int s, const int t,
  const std::complex<float>* RESTRICT A,
  const std::complex<float>* RESTRICT tau,
        Dense<std::complex<float> >& B,
        std::complex<float>* RESTRICT work );
template void ApplyPackedQAdjointFromLeft
( const int r, const int s, const int t,
  const std::complex<double>* RESTRICT A,
  const std::complex<double>* RESTRICT tau,
        Dense<std::complex<double> >& B,
        std::complex<double>* RESTRICT work );

template void ApplyPackedQFromRight
( const int r, const int s, const int t,
  const float* RESTRICT A,
  const float* RESTRICT tau,
        Dense<float>& B,
        float* RESTRICT work );
template void ApplyPackedQFromRight
( const int r, const int s, const int t,
  const double* RESTRICT A,
  const double* RESTRICT tau,
        Dense<double>& B,
        double* RESTRICT work );
template void ApplyPackedQFromRight
( const int r, const int s, const int t,
  const std::complex<float>* RESTRICT A,
  const std::complex<float>* RESTRICT tau,
        Dense<std::complex<float> >& B,
        std::complex<float>* RESTRICT work );
template void ApplyPackedQFromRight
( const int r, const int s, const int t,
  const std::complex<double>* RESTRICT A,
  const std::complex<double>* RESTRICT tau,
        Dense<std::complex<double> >& B,
        std::complex<double>* RESTRICT work );

template void ApplyPackedQAdjointFromRight
( const int r, const int s, const int t,
  const float* RESTRICT A,
  const float* RESTRICT tau,
        Dense<float>& B,
        float* RESTRICT work );
template void ApplyPackedQAdjointFromRight
( const int r, const int s, const int t,
  const double* RESTRICT A,
  const double* RESTRICT tau,
        Dense<double>& B,
        double* RESTRICT work );
template void ApplyPackedQAdjointFromRight
( const int r, const int s, const int t,
  const std::complex<float>* RESTRICT A,
  const std::complex<float>* RESTRICT tau,
        Dense<std::complex<float> >& B,
        std::complex<float>* RESTRICT work );
template void ApplyPackedQAdjointFromRight
( const int r, const int s, const int t,
  const std::complex<double>* RESTRICT A,
  const std::complex<double>* RESTRICT tau,
        Dense<std::complex<double> >& B,
        std::complex<double>* RESTRICT work );

template void PrintPacked
( const std::string msg,
  const int r, const int s, const int t, const float* packedA,
  std::ostream& os );
template void PrintPacked
( const std::string msg,
  const int r, const int s, const int t, const double* packedA,
  std::ostream& os );
template void PrintPacked
( const std::string msg,
  const int r, const int s, const int t, const std::complex<float>* packedA,
  std::ostream& os );
template void PrintPacked
( const std::string msg,
  const int r, const int s, const int t, const std::complex<double>* packedA,
  std::ostream& os );

} // namespace hmat_tools
} // namespace hifde3d

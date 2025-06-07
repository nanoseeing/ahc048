


// =========================================================
// Common
// =========================================================
#include <bits/stdc++.h>
using namespace std;

#include <boost/format.hpp>

// Judge環境切り替え
#ifndef ONLINE_JUDGE
#define _GLIBCXX_DEBUG
#include <cpp-dump.hpp>
#else
#define cpp_dump(...) ;
#endif

using ll = long long;
using Color = array<double, 3>;
using Fractor = pair<int, int>;
using Fractors = vector<Fractor>;

#define ALL(obj)  (obj).begin(), (obj).end()
#define RALL(obj) (obj).rbegin(), (obj).rend()
/* Non-Negagive Least Squares Algorithm for Eigen.
 *
 * Copyright (C) 2021 Essex Edwards, <essex.edwards@gmail.com>
 * Copyright (C) 2013 Hannes Matuschek, hannes.matuschek at uni-potsdam.de
 *
 * This Source Code Form is subject to the terms of the Mozilla
 * Public License v. 2.0. If a copy of the MPL was not distributed
 * with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

/** \defgroup nnls Non-Negative Least Squares (NNLS) Module
 * This module provides a single class @c Eigen::NNLS implementing the NNLS algorithm.
 * The algorithm is described in "SOLVING LEAST SQUARES PROBLEMS", by Charles L. Lawson and
 * Richard J. Hanson, Prentice-Hall, 1974 and solves optimization problems of the form
 *
 * \f[ \min \left\Vert Ax-b\right\Vert_2^2\quad s.t.\, x\ge 0\,.\f]
 *
 * The algorithm solves the constrained least-squares problem above by iteratively improving
 * an estimate of which constraints are active (elements of \f$x\f$ equal to zero)
 * and which constraints are inactive (elements of \f$x\f$ greater than zero).
 * Each iteration, an unconstrained linear least-squares problem solves for the
 * components of \f$x\f$ in the (estimated) inactive set and the sets are updated.
 * The unconstrained problem minimizes \f$\left\Vert A^Nx^N-b\right\Vert_2^2\f$,
 * where \f$A^N\f$ is a matrix formed by selecting all columns of A which are
 * in the inactive set \f$N\f$.
 *
 */

#ifndef EIGEN_NNLS_H
#define EIGEN_NNLS_H

#include <Eigen/Core>
#include <Eigen/QR>
#include <limits>

// !埋め込み
namespace Eigen {
namespace internal {

template <typename MatrixQR, typename HCoeffs, typename VectorQR>
void householder_qr_inplace_update(MatrixQR& mat, HCoeffs& hCoeffs, const VectorQR& newColumn, typename MatrixQR::Index k,
                                   typename MatrixQR::Scalar* tempData) {
    typedef typename MatrixQR::Index Index;
    typedef typename MatrixQR::RealScalar RealScalar;
    Index rows = mat.rows();

    eigen_assert(k < mat.cols());
    eigen_assert(k < rows);
    eigen_assert(hCoeffs.size() == mat.cols());
    eigen_assert(newColumn.size() == rows);
    eigen_assert(tempData);

    // Store new column in mat at column k
    mat.col(k) = newColumn;
    // Apply H = H_1...H_{k-1} on newColumn (skip if k=0)
    for(Index i = 0; i < k; ++i) {
        Index remainingRows = rows - i;
        mat.col(k).tail(remainingRows).applyHouseholderOnTheLeft(mat.col(i).tail(remainingRows - 1), hCoeffs.coeffRef(i), tempData + i + 1);
    }
    // Construct Householder projector in-place in column k
    RealScalar beta;
    mat.col(k).tail(rows - k).makeHouseholderInPlace(hCoeffs.coeffRef(k), beta);
    mat.coeffRef(k, k) = beta;
}

} // namespace internal
} // namespace Eigen

namespace Eigen {

/** \ingroup nnls
 * \class NNLS
 * \brief Implementation of the Non-Negative Least Squares (NNLS) algorithm.
 * \tparam MatrixType The type of the system matrix \f$A\f$.
 *
 * This class implements the NNLS algorithm as described in "SOLVING LEAST SQUARES PROBLEMS",
 * Charles L. Lawson and Richard J. Hanson, Prentice-Hall, 1974. This algorithm solves a least
 * squares problem iteratively and ensures that the solution is non-negative. I.e.
 *
 * \f[ \min \left\Vert Ax-b\right\Vert_2^2\quad s.t.\, x\ge 0 \f]
 *
 * The algorithm solves the constrained least-squares problem above by iteratively improving
 * an estimate of which constraints are active (elements of \f$x\f$ equal to zero)
 * and which constraints are inactive (elements of \f$x\f$ greater than zero).
 * Each iteration, an unconstrained linear least-squares problem solves for the
 * components of \f$x\f$ in the (estimated) inactive set and the sets are updated.
 * The unconstrained problem minimizes \f$\left\Vert A^Nx^N-b\right\Vert_2^2\f$,
 * where \f$A^N\f$ is a matrix formed by selecting all columns of A which are
 * in the inactive set \f$N\f$.
 *
 * See <a href="https://en.wikipedia.org/wiki/Non-negative_least_squares">the
 * wikipedia page on non-negative least squares</a> for more background information.
 *
 * \note Please note that it is possible to construct an NNLS problem for which the
 *       algorithm does not converge. In practice these cases are extremely rare.
 */
template <class MatrixType_>
class NNLS {
  public:
    typedef MatrixType_ MatrixType;

    enum {
        RowsAtCompileTime = MatrixType::RowsAtCompileTime,
        ColsAtCompileTime = MatrixType::ColsAtCompileTime,
        Options = MatrixType::Options,
        MaxRowsAtCompileTime = MatrixType::MaxRowsAtCompileTime,
        MaxColsAtCompileTime = MatrixType::MaxColsAtCompileTime
    };

    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::RealScalar RealScalar;
    typedef typename MatrixType::Index Index;

    /** Type of a row vector of the system matrix \f$A\f$. */
    typedef Matrix<Scalar, ColsAtCompileTime, 1> SolutionVectorType;
    /** Type of a column vector of the system matrix \f$A\f$. */
    typedef Matrix<Scalar, RowsAtCompileTime, 1> RhsVectorType;
    typedef Matrix<Index, ColsAtCompileTime, 1> IndicesType;

    /** */
    NNLS();

    /** \brief Constructs a NNLS sovler and initializes it with the given system matrix @c A.
     * \param A Specifies the system matrix.
     * \param max_iter Specifies the maximum number of iterations to solve the system.
     * \param tol Specifies the precision of the optimum.
     *        This is an absolute tolerance on the gradient of the Lagrangian, \f$A^T(Ax-b)-\lambda\f$
     *        (with Lagrange multipliers \f$\lambda\f$).
     */
    NNLS(const MatrixType &A, Index max_iter = -1, Scalar tol = NumTraits<Scalar>::dummy_precision());

    /** Initializes the solver with the matrix \a A for further solving NNLS problems.
     *
     * This function mostly initializes/computes the preconditioner. In the future
     * we might, for instance, implement column reordering for faster matrix vector products.
     */
    template <typename MatrixDerived>
    NNLS<MatrixType> &compute(const EigenBase<MatrixDerived> &A);

    /** \brief Solves the NNLS problem.
     *
     * The dimension of @c b must be equal to the number of rows of @c A, given to the constructor.
     *
     * \returns The approximate solution vector \f$ x \f$. Use info() to determine if the solve was a success or not.
     * \sa info()
     */
    const SolutionVectorType &solve(const RhsVectorType &b);

    /** \brief Returns the solution if a problem was solved.
     * If not, an uninitialized vector may be returned. */
    const SolutionVectorType &x() const {
        return x_;
    }

    /** \returns the tolerance threshold used by the stopping criteria.
     * \sa setTolerance()
     */
    Scalar tolerance() const {
        return tolerance_;
    }

    /** Sets the tolerance threshold used by the stopping criteria.
     *
     * This is an absolute tolerance on the gradient of the Lagrangian, \f$A^T(Ax-b)-\lambda\f$
     * (with Lagrange multipliers \f$\lambda\f$).
     */
    NNLS<MatrixType> &setTolerance(const Scalar &tolerance) {
        tolerance_ = tolerance;
        return *this;
    }

    /** \returns the max number of iterations.
     * It is either the value set by setMaxIterations or, by default, twice the number of columns of the matrix.
     */
    Index maxIterations() const {
        return max_iter_ < 0 ? 2 * A_.cols() : max_iter_;
    }

    /** Sets the max number of iterations.
     * Default is twice the number of columns of the matrix.
     * The algorithm requires at least k iterations to produce a solution vector with k non-zero entries.
     */
    NNLS<MatrixType> &setMaxIterations(Index maxIters) {
        max_iter_ = maxIters;
        return *this;
    }

    /** \returns the number of iterations (least-squares solves) performed during the last solve */
    Index iterations() const {
        return iterations_;
    }

    /** \returns Success if the iterations converged, and an error values otherwise. */
    ComputationInfo info() const {
        return info_;
    }

  private:
    /** \internal Adds the given index @c idx to the inactive set N and updates the QR decomposition of \f$A^N\f$. */
    void moveToInactiveSet_(Index idx);

    /** \internal Removes the given index idx from the inactive set N and updates the QR decomposition of \f$A^N\f$. */
    void moveToActiveSet_(Index idx);

    /** \internal Solves the least-squares problem \f$\left\Vert y-A^Nx\right\Vert_2^2\f$. */
    void solveInactiveSet_(const RhsVectorType &b);

  private:
    typedef Matrix<Scalar, ColsAtCompileTime, ColsAtCompileTime> MatrixAtAType;

    /** \internal Holds the maximum number of iterations for the NNLS algorithm.
     *  @c -1 means to use the default value. */
    Index max_iter_;
    /** \internal Holds the number of iterations. */
    Index iterations_;
    /** \internal Holds success/fail of the last solve. */
    ComputationInfo info_;
    /** \internal Size of the inactive set. */
    Index numInactive_;
    /** \internal Accuracy of the algorithm w.r.t the optimality of the solution (gradient). */
    Scalar tolerance_;
    /** \internal The system matrix, a copy of the one given to the constructor. */
    MatrixType A_;
    /** \internal Precomputed product \f$A^TA\f$. */
    MatrixAtAType AtA_;
    /** \internal Will hold the solution. */
    SolutionVectorType x_;
    /** \internal Will hold the current gradient.\f$A^Tb - A^TAx\f$ */
    SolutionVectorType gradient_;
    /** \internal Will hold the partial solution. */
    SolutionVectorType y_;
    /** \internal Precomputed product \f$A^Tb\f$. */
    SolutionVectorType Atb_;
    /** \internal Holds the current permutation partitioning the active and inactive sets.
     * The first @c numInactive_ elements form the inactive set and the rest the active set. */
    IndicesType index_sets_;
    /** \internal QR decomposition to solve the (inactive) sub system (together with @c qrCoeffs_). */
    MatrixType QR_;
    /** \internal QR decomposition to solve the (inactive) sub system (together with @c QR_). */
    SolutionVectorType qrCoeffs_;
    /** \internal Some workspace for QR decomposition. */
    SolutionVectorType tempSolutionVector_;
    RhsVectorType tempRhsVector_;
};

/* ********************************************************************************************
 * Implementation
 * ******************************************************************************************** */

template <typename MatrixType>
NNLS<MatrixType>::NNLS()
    : max_iter_(-1), iterations_(0), info_(ComputationInfo::InvalidInput), numInactive_(0), tolerance_(NumTraits<Scalar>::dummy_precision()) {
}

template <typename MatrixType>
NNLS<MatrixType>::NNLS(const MatrixType &A, Index max_iter, Scalar tol) : max_iter_(max_iter), tolerance_(tol) {
    compute(A);
}

template <typename MatrixType>
template <typename MatrixDerived>
NNLS<MatrixType> &NNLS<MatrixType>::compute(const EigenBase<MatrixDerived> &A) {
    // Ensure Scalar type is real. The non-negativity constraint doesn't obviously extend to complex numbers.
    EIGEN_STATIC_ASSERT(!NumTraits<Scalar>::IsComplex, NUMERIC_TYPE_MUST_BE_REAL);

    // max_iter_: unchanged
    iterations_ = 0;
    info_ = ComputationInfo::Success;
    numInactive_ = 0;
    // tolerance: unchanged
    A_ = A.derived();
    AtA_.noalias() = A_.transpose() * A_;
    x_.resize(A_.cols());
    gradient_.resize(A_.cols());
    y_.resize(A_.cols());
    Atb_.resize(A_.cols());
    index_sets_.resize(A_.cols());
    QR_.resize(A_.rows(), A_.cols());
    qrCoeffs_.resize(A_.cols());
    tempSolutionVector_.resize(A_.cols());
    tempRhsVector_.resize(A_.rows());

    return *this;
}

template <typename MatrixType>
const typename NNLS<MatrixType>::SolutionVectorType &NNLS<MatrixType>::solve(const RhsVectorType &b) {
    // Initialize solver
    iterations_ = 0;
    info_ = ComputationInfo::NumericalIssue;
    x_.setZero();

    index_sets_ = IndicesType::LinSpaced(A_.cols(), 0, A_.cols() - 1); // Identity permutation.
    numInactive_ = 0;

    // Precompute A^T*b
    Atb_.noalias() = A_.transpose() * b;

    const Index maxIterations = this->maxIterations();

    // OUTER LOOP
    while(true) {
        // Early exit if all variables are inactive, which breaks 'maxCoeff' below.
        if(A_.cols() == numInactive_) {
            info_ = ComputationInfo::Success;
            return x_;
        }

        // Find the maximum element of the gradient in the active set.
        // If it is small or negative, then we have converged.
        // Else, we move that variable to the inactive set.
        gradient_.noalias() = Atb_ - AtA_ * x_;

        const Index numActive = A_.cols() - numInactive_;
        Index argmaxGradient = -1;
        const Scalar maxGradient = gradient_(index_sets_.tail(numActive)).maxCoeff(&argmaxGradient);
        argmaxGradient += numInactive_; // because tail() skipped the first numInactive_ elements

        if(maxGradient < tolerance_) {
            info_ = ComputationInfo::Success;
            return x_;
        }

        moveToInactiveSet_(argmaxGradient);

        // INNER LOOP
        while(true) {
            // Check if max. number of iterations is reached
            if(iterations_ >= maxIterations) {
                info_ = ComputationInfo::NoConvergence;
                return x_;
            }

            // Solve least-squares problem in inactive set only,
            // this step is rather trivial as moveToInactiveSet_ & moveToActiveSet_
            // updates the QR decomposition of inactive columns A^N.
            // solveInactiveSet_ puts the solution in y_
            solveInactiveSet_(b);
            ++iterations_; // The solve is expensive, so that is what we count as an iteration.

            // Check feasibility...
            bool feasible = true;
            Scalar alpha = NumTraits<Scalar>::highest();
            Index infeasibleIdx = -1; // Which variable became infeasible first.
            for(Index i = 0; i < numInactive_; i++) {
                Index idx = index_sets_[i];
                if(y_(idx) < 0) {
                    // t should always be in [0,1].
                    Scalar t = -x_(idx) / (y_(idx) - x_(idx));
                    if(alpha > t) {
                        alpha = t;
                        infeasibleIdx = i;
                        feasible = false;
                    }
                }
            }
            eigen_assert(feasible || 0 <= infeasibleIdx);

            // If solution is feasible, exit to outer loop
            if(feasible) {
                x_ = y_;
                break;
            }

            // Infeasible solution -> interpolate to feasible one
            for(Index i = 0; i < numInactive_; i++) {
                Index idx = index_sets_[i];
                x_(idx) += alpha * (y_(idx) - x_(idx));
            }

            // Remove these indices from the inactive set and update QR decomposition
            moveToActiveSet_(infeasibleIdx);
        }
    }
}

template <typename MatrixType>
void NNLS<MatrixType>::moveToInactiveSet_(Index idx) {
    // Update permutation matrix:
    std::swap(index_sets_(idx), index_sets_(numInactive_));
    numInactive_++;

    // Perform rank-1 update of the QR decomposition stored in QR_ & qrCoeff_
    internal::householder_qr_inplace_update(QR_, qrCoeffs_, A_.col(index_sets_(numInactive_ - 1)), numInactive_ - 1, tempSolutionVector_.data());
}

template <typename MatrixType>
void NNLS<MatrixType>::moveToActiveSet_(Index idx) {
    // swap index with last inactive one & reduce number of inactive columns
    std::swap(index_sets_(idx), index_sets_(numInactive_ - 1));
    numInactive_--;
    // Update QR decomposition starting from the removed index up to the end [idx, ..., numInactive_]
    for(Index i = idx; i < numInactive_; i++) {
        Index col = index_sets_(i);
        internal::householder_qr_inplace_update(QR_, qrCoeffs_, A_.col(col), i, tempSolutionVector_.data());
    }
}

template <typename MatrixType>
void NNLS<MatrixType>::solveInactiveSet_(const RhsVectorType &b) {
    eigen_assert(numInactive_ > 0);

    tempRhsVector_ = b;

    // tmpRHS(0:numInactive_-1) := Q'*b
    // tmpRHS(numInactive_:end) := useless stuff we would rather not compute at all.
    tempRhsVector_.applyOnTheLeft(householderSequence(QR_.leftCols(numInactive_), qrCoeffs_.head(numInactive_)).transpose());

    // tempSol(0:numInactive_-1) := inv(R) * Q' * b
    //  = the least-squares solution for the inactive variables.
    tempSolutionVector_.head(numInactive_) =           //
        QR_.topLeftCorner(numInactive_, numInactive_)  //
            .template triangularView<Upper>()          //
            .solve(tempRhsVector_.head(numInactive_)); //

    // tempSol(numInactive_:end) := 0 = the value for the constrained variables.
    tempSolutionVector_.tail(y_.size() - numInactive_).setZero();

    // Back permute into original column order of A
    y_.noalias() = index_sets_.asPermutation() * tempSolutionVector_.head(y_.size());
}

} // namespace Eigen

#endif // EIGEN_NNLS_H
// Skipped: common.hpp already included

// =========================================================
// Utils
// =========================================================

// IO高速化
struct IOInit {
    IOInit() {
        ios::sync_with_stdio(0);
        cin.tie(0);
        cout << setprecision(15);
    }
} ioinit;

// 範囲for: [start, end) step
class range {
  public:
    class Iterator {
      public:
        using value_type = int;
        int value, step;

        template <integral T1, integral T2>
        Iterator(T1 value, T2 step) : value(value), step(step) {
        }

        auto operator*() const {
            return value;
        }

        Iterator &operator++() {
            value += step;
            return *this;
        }

        bool operator!=(const Iterator &other) const {
            return step > 0 ? value < other.value : value > other.value;
        }
    };

    template <integral T>
    range(T end) : range(0, end, 1) {
    }
    template <integral T1, integral T2>
    range(T1 start, T2 end) : range(start, end, 1) {
    }
    template <integral T1, integral T2, integral T3>
    range(T1 start, T2 end, T3 step) : begin_(start), end_(end), step_(step) {
        if(step == 0) {
            throw std::invalid_argument("Range step must not be 0");
        }
    }

    Iterator begin() const {
        return Iterator(begin_, step_);
    }
    Iterator end() const {
        return Iterator(end_, step_);
    }

  private:
    int begin_, end_, step_;
};

// ハッシュ（https://qiita.com/hamamu/items/4d081751b69aa3bb3557）
template <class T>
size_t HashCombine(const size_t seed, const T &v) {
    return seed ^ (std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}
template <class T, class S>
struct std::hash<std::pair<T, S>> {
    size_t operator()(const std::pair<T, S> &keyval) const noexcept {
        return HashCombine(std::hash<T>()(keyval.first), keyval.second);
    }
};
template <class T>
struct std::hash<std::vector<T>> {
    size_t operator()(const std::vector<T> &keyval) const noexcept {
        size_t s = 0;
        for(auto &&v : keyval)
            s = HashCombine(s, v);
        return s;
    }
};
template <int N>
struct HashTupleCore {
    template <class Tuple>
    size_t operator()(const Tuple &keyval) const noexcept {
        size_t s = HashTupleCore<N - 1>()(keyval);
        return HashCombine(s, std::get<N - 1>(keyval));
    }
};
template <>
struct HashTupleCore<0> {
    template <class Tuple>
    size_t operator()(const Tuple &keyval) const noexcept {
        return 0;
    }
};
template <class... Args>
struct std::hash<std::tuple<Args...>> {
    size_t operator()(const tuple<Args...> &keyval) const noexcept {
        return HashTupleCore<tuple_size<tuple<Args...>>::value>()(keyval);
    }
};

// Utils
template <typename T>
T intpow(T base, T exp, optional<T> mod = nullopt) {
    T result = 1;
    while(exp > 0) {
        if(exp & 1) {
            if(mod) {
                result = result * base % *mod;
            } else {
                result = result * base;
            }
        }
        exp >>= 1;
        if(exp <= 0) break;
        if(mod) {
            base = base * base % *mod;
        } else {
            base = base * base;
        }
    }
    return result;
}

template <typename T1, typename T2>
inline bool chmin(T1 &a, const T2 &b) {
    bool compare = a > b;
    if(a > b) a = b;
    return compare;
}
template <typename T1, typename T2>
inline bool chmax(T1 &a, const T2 &b) {
    bool compare = a < b;
    if(a < b) a = b;
    return compare;
}

// Set / Multiset
template <typename Set, typename T>
bool erase(Set &s, const T &x) {
    auto itr = s.find(x);
    if(itr != s.end()) {
        s.erase(itr);
        return true;
    }
    return false;
}

// queue / deque (コピーを返すので少しだけ処理が遅いのが不満。)
template <typename Q>
auto pop(Q &q) -> decltype(q.front(), void(), typename Q::value_type{}) {
    auto val = std::move(q.front());
    q.pop_front();
    return val;
}

// priority_queue (同上)
template <typename Q>
auto pop(Q &q) -> decltype(q.top(), void(), typename Q::value_type{}) {
    auto val = std::move(q.top());
    q.pop();
    return val;
}

class TimeKeeper {
  private:
    // high_resolution_clock → steady_clock に変更
    std::chrono::steady_clock::time_point start_time_;
    double time_threshold_;

  public:
    TimeKeeper(double time_threshold) : start_time_(std::chrono::steady_clock::now()), time_threshold_(time_threshold) {
    }

    double getElapsedTime() const {
        auto diff = std::chrono::steady_clock::now() - start_time_;
        return std::chrono::duration<double, std::milli>(diff).count();
    }

    bool isTimeOver() const {
        return getElapsedTime() >= time_threshold_;
    }
};

template <typename Derived, typename UIntType>
class XorshiftBase {
  public:
    using UInt = UIntType;

    UInt next() {
        return static_cast<Derived *>(this)->next();
    }

    // 任意の整数型を返すようテンプレート化（戻り値型を明示）
    UInt randint(UInt max) {
        return next() % max;
    }

    UInt randint(UInt low, UInt high) {
        return low + next() % (high - low + 1);
    }

    double rand() {
        constexpr int bits = std::numeric_limits<UInt>::digits;         // 仮数部のbit数ではなく、整数としてのbit数
        constexpr int float_bits = std::numeric_limits<double>::digits; // 仮数部の精度bit数（float=24, double=53）

        if constexpr(bits >= float_bits) {
            UInt value = next() >> (bits - float_bits); // 上位 float_bits を使う
            return static_cast<double>(value) / static_cast<double>(UInt(1) << float_bits);
        } else {
            return static_cast<double>(next()) / static_cast<double>(std::numeric_limits<UInt>::max());
        }
    }

    // 離散分布サンプリング（常に int でOK）
    int sample_discrete(const std::vector<double> &weights) {
        double total = std::accumulate(weights.begin(), weights.end(), 0.0);
        double r = rand() * total;
        double cumulative = 0.0;
        for(size_t i = 0; i < weights.size(); ++i) {
            cumulative += weights[i];
            if(r < cumulative) {
                return static_cast<int>(i);
            }
        }
        return static_cast<int>(weights.size() - 1);
    }

    // イテレータから k 個サンプル（順序ランダム）
    template <typename Iterator>
    std::vector<typename std::iterator_traits<Iterator>::value_type> random_sample(Iterator begin, Iterator end, int k) {
        using T = typename std::iterator_traits<Iterator>::value_type;
        std::vector<T> pool(begin, end);
        int n = static_cast<int>(pool.size());
        for(int i = 0; i < k; ++i) {
            int j = i + randint(n - i);
            std::swap(pool[i], pool[j]);
        }
        return std::vector<T>(pool.begin(), pool.begin() + k);
    }

    // シャッフル
    template <typename T>
    void shuffle(std::vector<T> &vec) {
        for(int i = (int)(vec.size()) - 1; i > 0; --i) {
            int j = randint(i + 1);
            std::swap(vec[i], vec[j]);
        }
    }
};

class Xorshift32 : public XorshiftBase<Xorshift32, uint32_t> {
  private:
    uint32_t state;

  public:
    explicit Xorshift32(uint32_t seed = 2525) : state(seed) {
    }

    uint32_t next() {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }
};

class Xorshift64 : public XorshiftBase<Xorshift64, uint64_t> {
  private:
    uint64_t state;

  public:
    explicit Xorshift64(uint64_t seed = 202520252025ULL) : state(seed) {
    }

    uint64_t next() {
        uint64_t x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state = x;
        return x;
    }
};

// 直積を生成する
template <typename T, typename Func>
void cartesian_product(const std::vector<std::vector<T>> &vectors, Func callback) {
    int n = vectors.size();
    std::vector<int> indices(n, 0);
    std::vector<T> result(n);

    while(true) {
        for(int i = 0; i < n; ++i) {
            result[i] = vectors[i][indices[i]];
        }
        callback(result); // ラムダが自動的に推論される

        int k = n - 1;
        while(k >= 0) {
            indices[k]++;
            if(indices[k] < static_cast<int>(vectors[k].size())) break;
            indices[k] = 0;
            --k;
        }
        if(k < 0) break;
    }
}

double exponential_schedule(double init, double obj, double elapsed_time, double max_time) {
    double lambda_param = log(obj / init) / max_time;
    return init * exp(lambda_param * elapsed_time);
}

double linear_schedule(double init, double obj, double elapsed_time, double max_time) {
    return init + (obj - init) * (elapsed_time / max_time);
}

pair<int, int> reduce_fraction(pair<int, int> frac) {
    int num = frac.first;
    int den = frac.second;

    if(den == 0) throw invalid_argument("Denominator cannot be zero");

    int g = gcd(abs(num), abs(den));
    num /= g;
    den /= g;

    return {num, den};
}

pair<int, int> mul_fracs(vector<pair<int, int>> fracs) {
    int num = 1;
    int den = 1;
    for(const auto &frac : fracs) {
        num *= frac.first;
        den *= frac.second;
    }
    return reduce_fraction({num, den});
}

template <typename RefT>
std::vector<size_t> make_sorted_indices(const std::vector<RefT> &ref, bool descending = false) {
    std::vector<size_t> indices(ref.size());
    for(size_t i = 0; i < ref.size(); ++i)
        indices[i] = i;

    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) { return descending ? ref[i] > ref[j] : ref[i] < ref[j]; });

    return indices;
}

template <typename T>
void reorder_vector(std::vector<T> &vec, const std::vector<size_t> &indices) {
    std::vector<T> reordered(vec.size());
    for(size_t i = 0; i < indices.size(); ++i) {
        reordered[i] = vec[indices[i]];
    }
    vec = std::move(reordered);
}

void choose_front(int start, int needed, int m, std::vector<int> &sel, std::vector<std::vector<int>> &result_list) {
    if(needed == 0) {
        std::vector<int> full = sel;
        full.push_back(m);
        result_list.push_back(full);
        return;
    }

    for(int i = start; i <= m - needed; ++i) {
        sel.push_back(i);
        choose_front(i + 1, needed - 1, m, sel, result_list);
        sel.pop_back();
    }
}

vector<vector<int>> choose_nCk(const int N, const int K, int max_comb = 10000) {
    std::vector<int> buffer;
    std::vector<std::vector<int>> tmp;
    vector<vector<int>> comb_list;
    for(int m = K - 1; m < N; ++m) {
        tmp.clear();
        buffer.clear();
        choose_front(0, K - 1, m, buffer, tmp);
        for(auto &comb : tmp) {
            if((int)comb_list.size() >= max_comb) {
                return comb_list;
            }
            comb_list.push_back(comb);
        }
    }

    return comb_list;
}

Xorshift64 xor_rng;
// ====================================
// NNLSを解くためのクラス
// ====================================

std::mt19937 engine(42);

#include <Eigen/Core>
#include <Eigen/Dense>

// 単純体への射影関数
Eigen::VectorXd ProjectOntoSimplex(const Eigen::VectorXd& v) {
    const int n = v.size();
    std::vector<double> u(n);
    for(int i = 0; i < n; ++i)
        u[i] = v[i];
    std::sort(u.begin(), u.end(), std::greater<double>());

    std::vector<double> cumsum(n);
    cumsum[0] = u[0];
    for(int i = 1; i < n; ++i)
        cumsum[i] = cumsum[i - 1] + u[i];

    int rho = -1;
    double theta = 0;
    for(int j = 0; j < n; ++j) {
        double t = (cumsum[j] - 1.0) / (j + 1);
        if(u[j] - t > 0) {
            rho = j;
            theta = t;
        }
    }
    if(rho < 0) {
        return Eigen::VectorXd::Constant(n, 1.0 / n);
    }
    Eigen::VectorXd w(n);
    for(int i = 0; i < n; ++i) {
        w[i] = std::max(v[i] - theta, 0.0);
    }
    return w;
}

vector<vector<int>> construct_subsets(int size, int k) {
    vector<vector<int>> subsets;
    vector<int> comb(size);
    function<void(int, int)> dfs = [&](int start, int depth) {
        if(depth == size) {
            subsets.emplace_back(comb.begin(), comb.end());
            return;
        }
        for(int x = start; x < k; x++) {
            comb[depth] = x;
            dfs(x + 1, depth + 1);
        }
    };
    dfs(0, 0);

    return subsets;
}

class ColorMixer {
  public:
    struct Result {
        double err;
        vector<int> indices;
        vector<double> weights;

        bool operator<(Result const& o) const {
            return err < o.err;
        }
    };

    struct SubsetInfo {
        int size;
        vector<int> indices;
    };

    static constexpr double EPS = 1e-7;
    static constexpr int MAX_ITER = 30;

    const int THRESHOLD = 500; // 20C3

    vector<Color> paints;
    int K;

    unordered_map<int, vector<vector<int>>> subsets_cache;
    ColorMixer(const vector<Color>& paints_input) : paints(paints_input) {
        K = paints.size();

        for(int i = 2; i <= 4; ++i) {
            auto subsets = construct_subsets(i, K);
            if((int)subsets.size() > THRESHOLD) {
                shuffle(subsets.begin(), subsets.end(), engine);
                subsets.resize(min((int)THRESHOLD, (int)subsets.size()));
            }
            subsets_cache[i] = move(subsets);
        }
    }

    double calc_true_error(vector<double>& weights, vector<int>& indices, Color& target) {
        double true_err = 0.0;
        for(int j = 0; j < 3; ++j) {
            double now_c = 0.0;
            for(int i = 0; i < (int)indices.size(); ++i) {
                int idx = indices[i];
                now_c += paints[idx][j] * weights[i];
            }
            double diff = now_c - target[j];
            true_err += diff * diff;
        }
        return sqrt(true_err);
    }

    Result nnls(Color& target, vector<int>& indices, double tol, double iter) {
        const int N = indices.size();

        Eigen::MatrixXd A_ext;
        A_ext.resize(4, N);
        for(int k = 0; k < N; ++k) {
            auto col = this->paints[indices[k]];
            Eigen::Vector3d c(col[0], col[1], col[2]);
            A_ext.block<3, 1>(0, k) = c;
        }
        A_ext.row(3).setOnes();

        Eigen::NNLS<Eigen::MatrixXd> nnls_solver;
        nnls_solver.compute(A_ext);
        nnls_solver.setTolerance(tol);
        nnls_solver.setMaxIterations(iter);

        Eigen::Vector4d t_ext;
        t_ext(0) = target[0];
        t_ext(1) = target[1];
        t_ext(2) = target[2];
        t_ext(3) = 1.0; // 「和が１になる」項を擬似的に加える

        Eigen::VectorXd x = nnls_solver.solve(t_ext);
        x = ProjectOntoSimplex(x); // 射影して非負かつ合計が1にする

        double sum_w = x.sum();
        assert(abs(sum_w - 1.0) < 1e-6);

        vector<double> weights;
        for(int i = 0; i < N; ++i) {
            weights.push_back(x(i) / sum_w); // 射影すれば1になるはずだが念のため正規化しておく
        }

        double true_err = calc_true_error(weights, indices, target);
        return Result{true_err, indices, weights};
    }

    vector<Result> solve_nnls(Color& t, int comb_size, int find_top_n) {
        assert(comb_size <= 4 && comb_size >= 2);

        vector<Result> results;

        if(comb_size == 4) { // !DEBUG
            // NNLSを解けば基本的に4色だけ残るはず。
            vector<int> indices;
            for(int i = 0; i < this->K; ++i) {
                indices.push_back(i);
            }
            Result r = nnls(t, indices, EPS, MAX_ITER);

            vector<int> inds4;
            vector<double> weights4;
            for(int i = 0; i < this->K; ++i) {
                if(r.weights[i] > EPS) {
                    inds4.push_back(i);
                    weights4.push_back(r.weights[i]);
                }
            }
            assert((int)inds4.size() <= 4);

            // 一応計算Errorは計算しなおさないといけないが、ほぼ誤差の範囲のはず
            Result new_r = Result{r.err, move(inds4), move(weights4)};
            results.emplace_back(move(new_r));
        }

        // 2, 3色のNNLSを解く
        auto& subsets = subsets_cache[comb_size];
        if(subsets.size() > THRESHOLD) {
            shuffle(subsets.begin(), subsets.end(), engine);
            subsets.resize(min(THRESHOLD, (int)subsets.size()));
        }

        for(auto& indices : subsets) {
            Result r = nnls(t, indices, EPS, MAX_ITER);
            results.emplace_back(move(r));
        }
        sort(ALL(results), [&](auto& a, auto& b) { return a.err < b.err; });
        results.resize(min(find_top_n, (int)results.size()));
        return results;
    }
};
// Skipped: common.hpp already included
// Skipped: common.hpp already included
// Skipped: utils.hpp already included

// =========================================================
// Game
// =========================================================

struct Input {
    int N, K, H, T, D;
    vector<Color> own;
    vector<Color> target;
};

double eval_error(Color col, Color tgt) {
    return sqrt(pow(col[0] - tgt[0], 2) + pow(col[1] - tgt[1], 2) + pow(col[2] - tgt[2], 2));
}

Color mix(double v1, Color c1, double v2, Color c2) {
    double sum = v1 + v2;
    if(sum <= 0) return {0.0, 0.0, 0.0};
    return {(v1 * c1[0] + v2 * c2[0]) / sum, (v1 * c1[1] + v2 * c2[1]) / sum, (v1 * c1[2] + v2 * c2[2]) / sum};
}

Color mix(vector<double> &vols, vector<Color> &colors) {
    double sum = 0.0;
    Color result = {0.0, 0.0, 0.0};
    for(int i : range(vols.size())) {
        sum += vols[i];
        result[0] += vols[i] * colors[i][0];
        result[1] += vols[i] * colors[i][1];
        result[2] += vols[i] * colors[i][2];
    }
    if(sum <= 0) return {0.0, 0.0, 0.0};
    return {result[0] / sum, result[1] / sum, result[2] / sum};
}

class Wall {
  public:
    vector<vector<bool>> wall_h;
    vector<vector<bool>> wall_v;
    Wall() = default;
    Wall(const vector<vector<bool>> &wall_h, const vector<vector<bool>> &wall_v) {
        // check size
        int horizontal_h = wall_h.size();
        int horizontal_w = wall_h[0].size();
        int vertical_h = wall_v.size();
        int vertical_w = wall_v[0].size();
        assert(horizontal_h + 1 == horizontal_w);
        assert(vertical_h == vertical_w + 1);
        assert(horizontal_h == vertical_w);

        this->wall_h = wall_h;
        this->wall_v = wall_v;
    }

    void switch_h(int i, int j) {
        wall_h[i][j] = wall_h[i][j] ^ true;
    }

    void switch_v(int i, int j) {
        wall_v[i][j] = wall_v[i][j] ^ true;
    }
};

enum class ActionType {
    Add = 1,
    Deliver = 2,
    Discard = 3,
    Toggle = 4,
};

struct Action {
    ActionType type;
    int i, j, k;
    int i2, j2;

    static Action Add(int i, int j, int k) {
        return {ActionType::Add, i, j, k, 0, 0};
    }
    static Action Deliver(int i, int j) {
        return {ActionType::Deliver, i, j, 0, 0, 0};
    }
    static Action Discard(int i, int j) {
        return {ActionType::Discard, i, j, 0, 0, 0};
    }
    static Action Toggle(int i1, int j1, int i2, int j2) {
        return {ActionType::Toggle, i1, j1, 0, i2, j2};
    }

    string to_string() const {
        if(type == ActionType::Add) {
            return boost::str(boost::format("Add: (%d, %d, %d)") % i % j % k);
        } else if(type == ActionType::Deliver) {
            return boost::str(boost::format("Deliver: (%d, %d)") % i % j);
        } else if(type == ActionType::Discard) {
            return boost::str(boost::format("Discard: (%d, %d)") % i % j);
        } else if(type == ActionType::Toggle) {
            return boost::str(boost::format("Toggle: (%d, %d) <-> (%d, %d)") % i % j % i2 % j2);
        } else {
            throw runtime_error("Unknown ActionType!");
        }
    }

    string to_string_output() const {
        int typei = static_cast<int>(this->type);
        if(type == ActionType::Add) {
            return boost::str(boost::format("%d %d %d %d") % typei % i % j % k);
        } else if(type == ActionType::Deliver) {
            return boost::str(boost::format("%d %d %d") % typei % i % j);
        } else if(type == ActionType::Discard) {
            return boost::str(boost::format("%d %d %d") % typei % i % j);
        } else if(type == ActionType::Toggle) {
            return boost::str(boost::format("%d %d %d %d %d") % typei % i % j % i2 % j2);
        } else {
            throw runtime_error("Unknown ActionType!");
        }
    }
};

tuple<int, vector<vector<int>>, vector<int>> get_ids(Wall &wall) {
    // TODO 壁の差分だけを更新するようにしたい。
    int N = wall.wall_v.size();
    vector<vector<int>> ids(N, vector<int>(N, -1));
    int ID = 0;
    vector<int> caps;

    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            if(ids[i][j] != -1) continue;

            vector<pair<int, int>> stack = {{i, j}};
            ids[i][j] = ID;
            int cap = 0;

            while(!stack.empty()) {
                auto [ci, cj] = stack.back();
                stack.pop_back();
                cap++;

                if(cj + 1 < N && !wall.wall_v[ci][cj] && ids[ci][cj + 1] == -1) {
                    ids[ci][cj + 1] = ID;
                    stack.emplace_back(ci, cj + 1);
                }
                if(ci + 1 < N && !wall.wall_h[ci][cj] && ids[ci + 1][cj] == -1) {
                    ids[ci + 1][cj] = ID;
                    stack.emplace_back(ci + 1, cj);
                }
                if(cj > 0 && !wall.wall_v[ci][cj - 1] && ids[ci][cj - 1] == -1) {
                    ids[ci][cj - 1] = ID;
                    stack.emplace_back(ci, cj - 1);
                }
                if(ci > 0 && !wall.wall_h[ci - 1][cj] && ids[ci - 1][cj] == -1) {
                    ids[ci - 1][cj] = ID;
                    stack.emplace_back(ci - 1, cj);
                }
            }

            caps.emplace_back(cap);
            ID++;
        }
    }

    return {ID, ids, caps};
}

struct Paint {
    int id;
    int cap;
    double vol;
    Color color;
};

struct State {
    Input input;
    Wall wall;
    Wall init_wall;
    vector<vector<int>> ids;
    vector<Paint> paints;

    vector<Color> delivered;
    vector<Action> actions;

    int turn = 0;
    int add_cnt = 0;
    double error = 0.0;
    double discard = 0.0;
    int deliver_cnt = 0;
    int discard_cnt = 0;

    State(const Wall &init_wall, const Input &input) {
        this->input = input;
        this->wall = init_wall;
        this->init_wall = init_wall;
        auto [ID, ids, caps] = get_ids(wall);
        this->ids = ids;
        for(int id : range(ID)) {
            this->paints.push_back({id, caps[id], 0.0, {0.0, 0.0, 0.0}});
        }
    }

    tuple<int, int, int> get_score() const {
        int deliver_cost = input.D * max(0, this->add_cnt - deliver_cnt);
        int err_cost = (int)round(1e4 * this->error);
        int total_cost = 1 + deliver_cost + err_cost;
        return {deliver_cost, err_cost, total_cost};
    }

    Paint get_paint(int i, int j) const {
        int id = this->ids[i][j];
        return this->paints[id];
    }

    Paint get_paint(int id) const {
        return this->paints[id];
    }

    void apply_add(const Action &action) {
        this->add_cnt++;
        int id = this->ids[action.i][action.j];
        double w = static_cast<double>(this->paints[id].cap) - this->paints[id].vol;
        if(w < 1.0) {
            this->paints[id].color = mix(this->paints[id].vol, this->paints[id].color, w, input.own[action.k]);
            this->paints[id].vol = static_cast<double>(this->paints[id].cap);
            throw runtime_error(boost::str(boost::format("Error: Paint volume exceeds capacity, turn: %d)") % this->turn));
        } else {
            this->paints[id].color = mix(this->paints[id].vol, this->paints[id].color, 1.0, input.own[action.k]);
            this->paints[id].vol += 1.0;
        }
    }

    void apply_deliver(const Action &action) {
        this->deliver_cnt++;
        int id = this->ids[action.i][action.j];
        if((int)this->delivered.size() >= input.H) {
            throw runtime_error("Error: Too many deliveries.");
        };
        if(this->paints[id].vol < 1.0 - 1e-6) {
            throw runtime_error("Error: Not enough paint to deliver.");
        };
        Color col = this->paints[id].color;
        Color tgt = input.target[this->delivered.size()];
        this->error += eval_error(col, tgt);
        this->paints[id].vol = max(0.0, this->paints[id].vol - 1.0);
        this->delivered.emplace_back(col);
    }

    void apply_discard(const Action &action) {
        this->discard_cnt++;
        int id = this->ids[action.i][action.j];
        if(this->paints[id].vol < 1e-6) {
            throw runtime_error("Error: Not enough paint to discard.");
        };
        discard += min(1.0, this->paints[id].vol);
        this->paints[id].vol = max(0.0, this->paints[id].vol - 1.0);
    }

    void apply_toggle(const Action &action) {
        int i1 = action.i, j1 = action.j;
        int i2 = action.i2, j2 = action.j2;
        if(i1 == i2) {
            auto i = i1;
            auto j = min(j1, j2);
            this->wall.switch_v(i, j);
        } else {
            auto i = min(i1, i2);
            auto j = j1;
            this->wall.switch_h(i, j);
        }
        auto [ID, ids, caps] = get_ids(this->wall);
        if(this->ids[i1][j1] == this->ids[i2][j2] && ids[i1][j1] != ids[i2][j2]) {
            auto id1 = ids[i1][j1];
            auto id2 = ids[i2][j2];
            auto v = this->paints[this->ids[i1][j1]].vol;
            auto vols = vector<double>(ID, 0.0);
            auto colors = vector<Color>(ID, {0.0, 0.0, 0.0});
            for(int i : range(input.N)) {
                for(int j : range(input.N)) {
                    vols[ids[i][j]] = this->paints[this->ids[i][j]].vol;
                    colors[ids[i][j]] = this->paints[this->ids[i][j]].color;
                }
            }
            vols[id1] = v * (double)caps[id1] / (double)(caps[id1] + caps[id2]);
            vols[id2] = v * (double)caps[id2] / (double)(caps[id1] + caps[id2]);
            this->ids = ids;
            vector<Paint> new_paints(ID);
            for(int id : range(ID)) {
                new_paints[id] = {id, caps[id], vols[id], colors[id]};
            }
            this->paints = new_paints;
        } else if(this->ids[i1][j1] != this->ids[i2][j2] && ids[i1][j1] == ids[i2][j2]) {
            auto id = ids[i1][j1];
            auto id1 = this->ids[i1][j1];
            auto id2 = this->ids[i2][j2];
            auto v1 = this->paints[id1].vol;
            auto v2 = this->paints[id2].vol;
            auto c1 = this->paints[id1].color;
            auto c2 = this->paints[id2].color;
            auto vols = vector<double>(ID, 0.0);
            auto colors = vector<Color>(ID, {0.0, 0.0, 0.0});
            for(int i : range(input.N)) {
                for(int j : range(input.N)) {
                    vols[ids[i][j]] = this->paints[this->ids[i][j]].vol;
                    colors[ids[i][j]] = this->paints[this->ids[i][j]].color;
                }
            }
            vols[id] = v1 + v2;
            colors[id] = mix(v1, c1, v2, c2);
            this->ids = ids;
            vector<Paint> new_paints(ID);
            for(int id : range(ID)) {
                new_paints[id] = {id, caps[id], vols[id], colors[id]};
            }
            this->paints = new_paints;
        }
    }

    void apply(const Action &action) {
        if(turn >= input.T) {
            throw runtime_error("Error: Too many turns.");
        }

        this->turn++;
        this->actions.emplace_back(action);

        if(action.type == ActionType::Add) {
            this->apply_add(action);
        } else if(action.type == ActionType::Deliver) {
            this->apply_deliver(action);
        } else if(action.type == ActionType::Discard) {
            this->apply_discard(action);
        } else if(action.type == ActionType::Toggle) {
            this->apply_toggle(action);
        } else {
            throw runtime_error("Unknown action type.");
        }
    }

    void debug() {
        for(const auto &paint : this->paints) {
            if(paint.vol < 1e-6) continue; // 1g未満は表示しない
            cerr << boost::format("ID: %d, Cap: %d, Vol: %.2f, Color: (%.2f, %.2f, %.2f)") % paint.id % paint.cap % paint.vol % paint.color[0] %
                        paint.color[1] % paint.color[2]
                 << endl;
        }
    }
};// Skipped: common.hpp already included
// Skipped: game.hpp already included

// =========================================================
// IO
// =========================================================

struct Output {
    Wall init_wall;
    vector<Action> actions;
};

Input parse_input() {
    Input input;
    cin >> input.N >> input.K >> input.H >> input.T >> input.D;
    input.own.resize(input.K);
    for(int i = 0; i < input.K; ++i) {
        for(int j = 0; j < 3; ++j) {
            cin >> input.own[i][j];
        }
    }
    input.target.resize(input.H);
    for(int i = 0; i < input.H; ++i) {
        for(int j = 0; j < 3; ++j) {
            cin >> input.target[i][j];
        }
    }
    return input;
}

void print_output(Output &output) {
    const auto &wall = output.init_wall;
    for(int i = 0; i < (int)wall.wall_v.size(); ++i) {
        for(int j = 0; j < (int)wall.wall_v[i].size(); ++j) {
            cout << (wall.wall_v[i][j] ? "1" : "0") << " ";
        }
        cout << "\n";
    }
    for(int i = 0; i < (int)wall.wall_h.size(); ++i) {
        for(int j = 0; j < (int)wall.wall_h[i].size(); ++j) {
            cout << (wall.wall_h[i][j] ? "1" : "0") << " ";
        }
        cout << "\n";
    }

    for(const auto &action : output.actions) {
        cout << action.to_string_output() << "\n";
    }
}// Skipped: utils.hpp already included

// ============================================================================
// 定義
// ============================================================================

const double MAX_TIME = 2800.0;
const int INIT_PARTITION_POS = 1;                        // パーティション初期値
long long MAX_SIMULATE_CNT = 2e7;                        // 分数パターンの最大数（目安）
const int BUFFER_TURN = 10;                              // 念のためバッファを持たせる
const int SEARCH_LEFT = -1;                              // 直積の左側を探索
const int SEARCH_RIGHT = 1;                              // 直積の右側を探索
const double BUF_MUL_TURN = 2.0;                         // 色数 x 4.0 + BUF_MUL_TURNぐらい掛かるはず
const double SWITH_POLICY_OBJ_TURN = 8.0 + BUF_MUL_TURN; // 2色（8.0ターン）も混合できないなら、分数混合を諦める
const vector<pair<int, int>> COMB_SEARCH_NUMS = {{2, 3}, {3, 5}, {4, 25}};

// ============================================================================
// Main
// ============================================================================

struct GroupInfo {
    int k;
    int row_num;
    int start_x;
    std::vector<std::pair<int, int>> roots;
    int now_pos;
    int size;
};

class ColorGroupManager {
  private:
    int n;
    int k;
    int original_k;
    int init_pos;
    std::vector<GroupInfo> infos;

    std::vector<std::pair<int, int>> create_root(int x, int row_num) {
        std::vector<std::pair<int, int>> roots;
        roots.emplace_back(n - 1, x);

        for(int r = 0; r < row_num; ++r) {
            if(r % 2 == 0) {
                for(int i = n - 2; i >= 0; --i) {
                    roots.emplace_back(i, x + r);
                }
            } else {
                for(int i = 0; i < n - 1; ++i) {
                    roots.emplace_back(i, x + r);
                }
            }
        }

        std::reverse(roots.begin(), roots.end());
        return roots;
    }

    std::vector<GroupInfo> construct_group_info() {
        std::vector<int> num_list(k, 1);
        for(int i = 0; i < n - k; ++i) {
            num_list[0] += 1;
            std::sort(num_list.begin(), num_list.end());
        }

        std::vector<int> acc_num_list(k, 0);
        for(int i = 0; i < k - 1; ++i) {
            acc_num_list[i + 1] = num_list[i] + acc_num_list[i];
        }

        std::vector<GroupInfo> result;
        result.reserve(k);
        for(int ki = 0; ki < k; ++ki) {
            int row_num = num_list[ki];
            int start_x = acc_num_list[ki];
            auto roots = create_root(start_x, row_num);

            GroupInfo info;
            info.k = ki;
            info.row_num = row_num;
            info.start_x = start_x;
            info.roots = std::move(roots);
            info.now_pos = init_pos;
            info.size = static_cast<int>(info.roots.size()) - 1;
            result.push_back(std::move(info));
        }

        return result;
    }

  public:
    ColorGroupManager(int n_, int k_, int original_k_, int init_pos_ = 2) : n(n_), k(k_), original_k(original_k_), init_pos(init_pos_) {
        infos = construct_group_info();
    }

    vector<int> get_unique_sizes() {
        set<int> unique_denoms;
        for(int ki : range(this->k)) {
            unique_denoms.insert(this->get_size(ki));
        }
        vector<int> denoms(ALL(unique_denoms));
        return denoms;
    }

    int get_start_x(int k_index) const {
        return infos[k_index].start_x;
    }

    int get_now_pos(int k_index) const {
        return infos[k_index].now_pos;
    }

    int get_size(int k_index) const {
        return infos[k_index].size;
    }
    std::tuple<int, int, int, int> get_partition_pos(int k_index, int num) const {
        assert(0 < num && num <= infos[k_index].size);
        auto [y1, x1] = infos[k_index].roots[num - 1];
        auto [y2, x2] = infos[k_index].roots[num];
        return {y1, x1, y2, x2};
    }

    void change_now_pos(int k_index, int pos) {
        infos[k_index].now_pos = pos;
    }

    void apply_reserved_changes(vector<pair<int, int>> &reserved_changes) {
        for(auto &[k_index, pos] : reserved_changes) {
            this->change_now_pos(k_index, pos);
        }
    }

    Action get_toggle_action(int k_index, int num) const {
        auto [y1, x1, y2, x2] = this->get_partition_pos(k_index, num);
        return Action::Toggle(y1, x1, y2, x2);
    }

    Action get_add_paint_action(int k_index) const {
        auto [y, x] = infos[k_index].roots[0];
        return Action::Add(y, x, k_index % this->original_k);
    }

    Action get_deliver_paint_action(int k_index) const {
        auto [y, x] = infos[k_index].roots[0];
        return Action::Deliver(y, x);
    }

    Paint get_paint(int k_index, const State &state) const {
        auto [y, x] = infos[k_index].roots[0];
        auto paint = state.get_paint(y, x);
        return paint;
    }

    Wall struct_init_wall(Input &input_data) {
        vector<vector<bool>> wall_h(input_data.N - 1, vector<bool>(input_data.N, false));
        vector<vector<bool>> wall_v(input_data.N, vector<bool>(input_data.N - 1, false));

        for(int x : range(input_data.N - 1)) {
            for(int y : range(input_data.N - 1)) {
                wall_v[y][x] = true;
            }
        }
        for(int x : range(input_data.N)) {
            wall_h[input_data.N - 2][x] = true;
        }

        // ルート内の仕切りを外す
        for(int k : range(input_data.K)) {
            const int root_size = (int)infos[k].roots.size();
            for(int i : range(1, root_size)) {
                auto [y1, x1] = infos[k].roots[i - 1];
                auto [y2, x2] = infos[k].roots[i];
                if(y1 == y2) {
                    wall_v[y1][min(x1, x2)] = false;
                } else {
                    wall_h[min(y1, y2)][x1] = false;
                }
            }
        }

        // 初期のパーティション位置を設定
        for(int k : range(input_data.K)) {
            auto [y1, x1, y2, x2] = this->get_partition_pos(k, this->init_pos);
            if(y1 == y2) {
                wall_v[y1][min(x1, x2)] = true;
            } else {
                wall_h[min(y1, y2)][x1] = true;
            }
        }

        // 混合する仕切りを開けておく
        for(int k : range(input_data.K)) {
            int s = this->get_size(k);
            auto [y1, x1, y2, x2] = this->get_partition_pos(k, s);
            assert(x1 == x2);
            wall_h[min(y1, y2)][x1] = false;
        }

        return Wall(wall_h, wall_v);
    }
};

class FractorManager {
    using KEY = tuple<int, int, int>; // (init_pos, max_denom, apply_frac_cnt)

  private:
    unordered_map<KEY, vector<Fractors>> fractor_map;
    unordered_map<KEY, vector<double>> rates_map;

  public:
    FractorManager(vector<int> &max_denominators) {
        for(auto &max_denom : max_denominators) {
            this->construct(max_denom, 1);
            this->construct(max_denom, 2);
        }
    }

    double calc_rate(const Fractors &fractors) const {
        double rate = 1.0;
        for(auto &fractor : fractors) {
            if(fractor.first == -1) {
                rate = 0.0;
            } else {
                rate *= (double)(fractor.first) / (double)(fractor.second);
            }
        }
        return rate;
    }

    void construct(int max_denom, int apply_frac_cnt) {
        vector<Fractors> fractors;
        vector<double> rates;

        // 全開放
        fractors.push_back({make_pair(1, 1)});
        rates.push_back(1.0);

        // 何もしない
        fractors.push_back({make_pair(-1, -1)});
        rates.push_back(0.0);

        // !INFO
        // 分数の適応パターンが多すぎる場合、パターンの列挙だけでTLEする可能性がある。
        // 仕方なくパターン数を減らすが、もっと良い方法があるかもしれない。

        int MAX_STEP = 1;
        long long simulate_cnt = pow(max_denom, 5);
        if(simulate_cnt > MAX_SIMULATE_CNT) {
            double div = (double)simulate_cnt / (double)MAX_SIMULATE_CNT;
            MAX_STEP = max(1, int(round(pow(div, 1.0 / 5.0))));
        }

        unordered_set<Fractor> fractor_set;
        for(int init_pos : range(max_denom, 0, -1)) {
            for(int fractor_cnt : range(apply_frac_cnt)) {
                // 分数1回適応
                if(fractor_cnt == 0) {
                    for(int denominator : range(init_pos, max_denom + 1)) {
                        for(int numerator : range(1, denominator)) {
                            Fractor fractor = make_pair(numerator, denominator);
                            Fractor reduced_fractor = reduce_fraction(fractor);
                            if(fractor_set.contains(reduced_fractor)) continue;
                            fractor_set.insert(reduced_fractor);
                            fractors.push_back({fractor});
                            rates.emplace_back(calc_rate(fractors.back()));
                        }
                    }
                } else if(fractor_cnt == 1) {
                    // 分数2回適応
                    for(int d1 : range(init_pos + 1, max_denom + 1, MAX_STEP)) {
                        for(int n1 : range(1, d1, MAX_STEP)) {
                            Fractor f1 = make_pair(n1, d1);
                            // 次の分母の最小値 = 下側のブロック数 = 前の分子
                            // 次の分母の最大値 += 最大ブロック数 - 前の分母
                            for(int d2 : range(max(2, n1), n1 + (max_denom - d1) + 1, MAX_STEP)) {
                                for(int n2 : range(1, d2, MAX_STEP)) {
                                    Fractor f2 = make_pair(n2, d2);
                                    Fractor fractor = mul_fracs({f1, f2});
                                    if(fractor_set.contains(fractor)) continue;
                                    fractor_set.insert(fractor);
                                    fractors.push_back({f1, f2});
                                    rates.emplace_back(calc_rate(fractors.back()));
                                }
                            }
                        }
                    }
                }
            }

            // 2分探索のためソートしておく必要がある
            auto inds = make_sorted_indices(rates);
            reorder_vector(fractors, inds);
            reorder_vector(rates, inds);

            KEY key = make_tuple(init_pos, max_denom, apply_frac_cnt);
            fractor_map[key] = fractors;
            rates_map[key] = rates;
        }
    }

    const vector<Fractors> &get_fractors(int pos, int max_denom, int apply_frac_cnt) const {
        KEY key = make_tuple(pos, max_denom, apply_frac_cnt);
        return fractor_map.at(key);
    }

    const vector<double> &get_rates(int pos, int max_denom, int apply_frac_cnt) const {
        KEY key = make_tuple(pos, max_denom, apply_frac_cnt);
        return rates_map.at(key);
    }

    pair<double, Fractors> get(int pos, int max_denom, int apply_frac_cnt, int i) const {
        KEY key = make_tuple(pos, max_denom, apply_frac_cnt);
        auto &frac = fractor_map.at(key)[i];
        auto &rate = rates_map.at(key)[i];
        return {rate, frac};
    }
};

struct ImmediateInfo {
    int k;
    bool is_add;
    double rate;
    double vol;
    Fractors fractors;
};

struct DicisionAction {
    vector<Action> pre_actions;
    vector<Action> release_actions;
    vector<Action> post_actions;
    int act_cnt;
    int change_color_num;

    vector<pair<int, int>> reserved_changes; // (k_index, pos)

    double cost;
};

class PolicyFractor {
  public:
    Input &input;
    State &state;
    ColorMixer &mixer;
    ColorGroupManager &color_group_manager;
    FractorManager &fractor_manager;
    TimeKeeper &time_keeper;

    double start_time = 0.0; // TODO

    // コンストラクタ
    PolicyFractor(Input &input_, State &state_, ColorMixer &mixer_, ColorGroupManager &color_group_manager_, FractorManager &fractor_manager_,
                  TimeKeeper &time_keeper_)
        : input(input_), state(state_), mixer(mixer_), color_group_manager(color_group_manager_), fractor_manager(fractor_manager_), time_keeper(time_keeper_) {
    }

    tuple<int, int> search_target_weight_idx(int k, double target_vol, bool is_add, int max_mul_cnt) {
        double now_vol = color_group_manager.get_paint(k, this->state).vol;
        int now_pos = color_group_manager.get_now_pos(k);
        int max_group_size = color_group_manager.get_size(k);
        auto &rates = fractor_manager.get_rates(now_pos, max_group_size, max_mul_cnt);

        // ---------------------------------------
        // now_vol * rate = target_vol
        // rate = target_vol / now_vol
        // rate = target_vol / (now_vol + 1.0)
        // ---------------------------------------
        double search_rate;
        if(is_add) {
            search_rate = target_vol / (1.0 + now_vol);
        } else {
            search_rate = target_vol / now_vol;
        }
        auto it = upper_bound(ALL(rates), search_rate);
        int it_ind = distance(rates.begin(), it);
        int rates_size = (int)rates.size();
        if(it_ind >= rates_size) {
            it_ind = rates_size - 1;
        }
        return {it_ind, rates_size};
    }

    double eval_cost(vector<ImmediateInfo> &immeediate_info) {
        auto &now_target = this->state.input.target[this->state.deliver_cnt];

        double sum_vol = 0.0;
        int add_cnt = 0;
        vector<double> vols;
        vector<Color> colors;
        for(auto info : immeediate_info) {
            vols.emplace_back(info.vol);
            colors.emplace_back(this->input.own[info.k]);
            sum_vol += info.vol;
            if(info.is_add) add_cnt++;
        }

        Color mixed_color = mix(vols, colors);
        double err_cost = eval_error(mixed_color, now_target) * 1e4;
        double discard_cost = max(0.0, sum_vol - 1.0) * (double)(this->input.D);

        int total_add_cnt = this->state.add_cnt + add_cnt;
        if(total_add_cnt > input.H) {
            double add_cost = (total_add_cnt - input.H) * (double)(this->input.D);
            return err_cost + add_cost;
        } else {
            return err_cost + discard_cost;
        }
    }

    tuple<vector<ImmediateInfo>, double> eval_one_result(ColorMixer::Result &constrait, vector<int> &max_frac_cnt) {
        int comb_size = constrait.indices.size();

        // 2^comb_size 個の組み合わせを評価する
        vector<vector<ImmediateInfo>> infos;
        for(int comb_ind : range(comb_size)) {
            auto &k = constrait.indices[comb_ind];
            auto &target_vol = constrait.weights[comb_ind];
            double now_vol = color_group_manager.get_paint(k, state).vol;
            bool is_add = (target_vol > now_vol) ? true : false;
            auto [it_ind, max_ind] = search_target_weight_idx(k, target_vol, is_add, max_frac_cnt[comb_ind]);
            vector<ImmediateInfo> immediate_infos;

            for(int j : range(SEARCH_LEFT, SEARCH_RIGHT)) {
                if(it_ind + j < 0 || it_ind + j >= max_ind) continue;
                int new_ind = it_ind + j;
                auto [rate, fractors] =
                    fractor_manager.get(color_group_manager.get_now_pos(k), color_group_manager.get_size(k), max_frac_cnt[comb_ind], new_ind);
                double vol;
                if(is_add) {
                    vol = (now_vol + 1.0) * rate;
                } else {
                    vol = now_vol * rate;
                }
                ImmediateInfo info = {.k = k, .is_add = is_add, .rate = rate, .vol = vol, .fractors = fractors};
                immediate_infos.emplace_back(info);
            }
            infos.emplace_back(move(immediate_infos));
        }

        double best_cost = 1e9;
        vector<ImmediateInfo> best_info;
        cartesian_product(infos, [&](vector<ImmediateInfo> &comb) {
            double sum_vol = 0.0;
            for(const auto &info : comb) {
                sum_vol += info.vol;
            }
            if(sum_vol > 1.0 - 1e-6) {
                double cost = eval_cost(comb);
                if(cost < best_cost) {
                    best_cost = cost;
                    best_info = comb;
                }
            }
        });

        return {best_info, best_cost};
    }

    DicisionAction construct_from_immediateinfo(vector<ImmediateInfo> &best_info) {
        DicisionAction action_result;
        action_result.change_color_num = (int)best_info.size();
        vector<pair<int, int>> reserved_changes;

        for(auto &info : best_info) {
            int now_partition_pos = color_group_manager.get_now_pos(info.k);
            int frac_size = info.fractors.size();

            auto &first_fractor = info.fractors[0];
            if(first_fractor.first == -1 && first_fractor.second == -1) {
                // 何もしない
                continue;
            } else if(first_fractor.first == 1 && first_fractor.second == 1) {
                // 全開放
                assert(frac_size == 1);
                action_result.release_actions.emplace_back(color_group_manager.get_toggle_action(info.k, now_partition_pos));
                if(info.is_add) {
                    // 仕切りを解放してから絵の具追加する(release_act)
                    action_result.release_actions.emplace_back(color_group_manager.get_add_paint_action(info.k));
                }
                action_result.post_actions.emplace_back(color_group_manager.get_toggle_action(info.k, INIT_PARTITION_POS));
                reserved_changes.emplace_back(info.k, INIT_PARTITION_POS);
            } else {
                // 分割n回適応
                int upper_partition = 0;
                int lower_partition = now_partition_pos;
                for(int fi : range(frac_size)) {
                    auto &fractor = info.fractors[fi];

                    // 上の仕切りから、分母だけ進んだのがstopしたいしきり位置
                    int stop_par_pos = upper_partition + fractor.second;
                    // stopする仕切りから、分子だけ進んだのが、releaseする仕切り位置
                    int release_par_pos = stop_par_pos - fractor.first;

                    if(stop_par_pos != lower_partition) {
                        // 現在の仕切りを動かす必要があるなら、仕切りを拡張する
                        action_result.pre_actions.emplace_back(color_group_manager.get_toggle_action(info.k, stop_par_pos));
                        action_result.pre_actions.emplace_back(color_group_manager.get_toggle_action(info.k, lower_partition));
                        // 拡張した後に追加する
                        if(fi == 0 && info.is_add) {
                            action_result.pre_actions.emplace_back(color_group_manager.get_add_paint_action(info.k));
                        }
                    } else {
                        // 追加する
                        if(fi == 0 && info.is_add) {
                            assert(lower_partition > 1);
                            action_result.pre_actions.emplace_back(color_group_manager.get_add_paint_action(info.k));
                        }
                    }
                    // 分子の位置で止める
                    action_result.pre_actions.emplace_back(color_group_manager.get_toggle_action(info.k, release_par_pos));

                    if(fi == frac_size - 1) {
                        // 分母の位置で解放する
                        action_result.release_actions.emplace_back(color_group_manager.get_toggle_action(info.k, stop_par_pos));
                        // 最後の仕切り位置は、release地点になる
                        reserved_changes.emplace_back(info.k, release_par_pos);
                    } else {
                        // 仮止めした仕切りは解放しておく必要がある
                        action_result.post_actions.emplace_back(color_group_manager.get_toggle_action(info.k, release_par_pos));
                    }

                    upper_partition = release_par_pos;
                    lower_partition = stop_par_pos;
                }
            }
        }
        int act_cnt = action_result.pre_actions.size() + action_result.release_actions.size() + action_result.post_actions.size();
        action_result.act_cnt = act_cnt;
        action_result.reserved_changes = reserved_changes;

        return action_result;
    }

    DicisionAction dicision_action(double obj_turn) {
        // TODO
        if(this->start_time == 0.0) {
            this->start_time = this->time_keeper.getElapsedTime();
        }
        double elapsed_time = this->time_keeper.getElapsedTime() - this->start_time;

        Color target = input.target[state.deliver_cnt];

        double best_cost = 1e9;
        vector<ImmediateInfo> best_info;
        for(const auto &comb_search_num : COMB_SEARCH_NUMS) {
            auto [comb_size, now_search_num] = comb_search_num;
            double remain_turn = obj_turn - comb_size * 4.0 - BUF_MUL_TURN;
            if(remain_turn < 0.0) {
                continue; // 目標ターン数を超える場合はスキップ
            }
            auto results = mixer.solve_nnls(target, comb_size, now_search_num);
            for(auto &result : results) {
                int max_double_frac_num = (int)(remain_turn / 4.0); // 分数2回適応できる数
                max_double_frac_num = min(max_double_frac_num, comb_size);
                vector<int> max_frac_cnt(comb_size, 1);
                if(max_double_frac_num > 0) {
                    for(int i : range(max_double_frac_num)) {
                        max_frac_cnt[comb_size - i - 1] = 2;
                    }
                }
                do {
                    auto [now_info, now_cost] = this->eval_one_result(result, max_frac_cnt);
                    if(now_cost < best_cost) {
                        best_cost = now_cost;
                        best_info = now_info;
                    }
                } while(next_permutation(ALL(max_frac_cnt)));
            }
        }

        assert((int)best_info.size() != 0);

        auto action_result = construct_from_immediateinfo(best_info);
        action_result.cost = best_cost;
        return action_result;
    }
};
class PolicyGreedy {
  public:
    const int MAX_MIX_COLOR_NUM = 5; // N色まで混合可能にしておかないと、メモリが足りない
    Input &input;
    State &state;
    vector<Color> mix_cache;

    PolicyGreedy(Input &input, State &state) : input(input), state(state) {
        construct();
    }

    void construct() {
        mix_cache.resize(1 << input.K);
        for(int i : range(1, 1 << input.K)) {
            vector<Color> colors;
            vector<double> vols;

            for(int k : range(input.K)) {
                if((i >> k) & 1) {
                    colors.push_back(input.own[k]);
                    vols.push_back(1.0);
                }
            }

            Color mixed_color = mix(vols, colors);
            mix_cache[i] = mixed_color;
        }
    }

    DicisionAction dicision_action(double obj_turn) {
        auto target_color = input.target[state.deliver_cnt];
        int can_mixed_num = min(max(1, int(obj_turn / 2.0)), MAX_MIX_COLOR_NUM);

        double min_cost = 1e18;
        int min_ind = -1;
        std::vector<int> indices(input.K, 0);
        for(int mixed_size = 0; mixed_size <= can_mixed_num; ++mixed_size) {
            std::fill(indices.begin(), indices.end(), 0);
            std::fill(indices.begin(), indices.begin() + mixed_size, 1);
            do {
                int x = 0;
                for(int i = 0; i < input.K; ++i) {
                    if(indices[i]) x |= (1 << i);
                }
                if(x == 0) continue; // 少なくとも1色は選ぶ必要がある
                auto &mixed_color = mix_cache[x];

                double total_cost = 0.0;
                double error_cost = eval_error(mixed_color, target_color) * 1e4;
                int total_add_cnt = this->state.add_cnt + mixed_size;
                if(total_add_cnt > input.H) {
                    double add_cost = (total_add_cnt - input.H) * (double)(this->input.D);
                    total_cost = error_cost + add_cost;
                } else {
                    total_cost = error_cost + (double)(input.D) * (mixed_size - 1.0);
                }
                if(total_cost < min_cost) {
                    min_cost = total_cost;
                    min_ind = x;
                }
            } while(std::prev_permutation(indices.begin(), indices.end()));
        }

        vector<int> target_inds;
        for(int k = 0; k < input.K; ++k) {
            if((min_ind >> k) & 1) {
                target_inds.push_back(k);
            }
        }

        vector<Action> actions;
        for(const auto &k : target_inds) {
            actions.push_back(Action::Add(input.N - 1, 0, k));
        }

        DicisionAction action_result;
        action_result.pre_actions = actions;
        action_result.cost = min_cost;
        action_result.change_color_num = 99; // TODO: 特に意味がない
        return action_result;
    }
};

void print_info(State &state) {
    auto [deliver_cost, err_cost, total_cost] = state.get_score();
    cerr << boost::format("H: %4d | Turn: %5d/%5d | Add: %4d | Discard: %4d (%5d loss) | Score: %5d (add: %5d, err: %5d)") % state.deliver_cnt % state.turn %
                state.input.T % state.add_cnt % state.discard_cnt % int(state.discard * 1e4) % total_cost % deliver_cost % err_cost
         << endl;
}

void apply_actions(DicisionAction &dicision_act, State &state, Input &input, bool is_end) {
    for(const auto &act : dicision_act.pre_actions) {
        state.apply(act);
    }
    for(const auto &act : dicision_act.release_actions) {
        state.apply(act);
    }
    state.apply(Action::Deliver(input.N - 1, 0));

    if(is_end) return; // 最終ターンは配達したら終了

    while(state.get_paint(input.N - 1, 0).vol > 1e-6) {
        state.apply(Action::Discard(input.N - 1, 0));
    }
    for(const auto &act : dicision_act.post_actions) {
        state.apply(act);
    }
}

void solve() {
    TimeKeeper time_keeper(MAX_TIME);

    Input input = parse_input();
    ColorGroupManager color_group_manager(input.N, input.K, input.K, INIT_PARTITION_POS);
    auto unique_sizes_ = color_group_manager.get_unique_sizes();

    FractorManager fractor_manager(unique_sizes_);
    auto init_wall = color_group_manager.struct_init_wall(input);
    State state(init_wall, input);
    ColorMixer mixer(input.own);

    PolicyGreedy policy_greedy(input, state);
    PolicyFractor policy_fractor(input, state, mixer, color_group_manager, fractor_manager, time_keeper);

    // Main Loop
    int policy_greedy_cnt = 0;
    double policy_err_sum = 0.0;
    map<int, int> act_cnt;
    map<int, int> color_cnt;

    try {
        for(int h : range(input.H)) {
            if(h % 10 == 0) print_info(state);
            int remain_turn = input.T - state.turn - BUFFER_TURN;
            double obj_turn = (double)remain_turn / (double)(input.H - state.deliver_cnt);

            DicisionAction best_act;
            if(obj_turn >= SWITH_POLICY_OBJ_TURN) {
                best_act = policy_fractor.dicision_action(obj_turn);
            } else {
                best_act = policy_greedy.dicision_action(obj_turn);
                policy_greedy_cnt++;
                policy_err_sum += best_act.cost;
            }
            apply_actions(best_act, state, input, (state.deliver_cnt + 1) == input.H);
            color_group_manager.apply_reserved_changes(best_act.reserved_changes);

            act_cnt[best_act.change_color_num] += best_act.act_cnt;
            color_cnt[best_act.change_color_num]++;
        }
        print_info(state);
    } catch(const exception &e) {
        Output output = Output{init_wall, state.actions};
        print_output(output);
        cerr << "Exception: " << e.what() << endl;
        exit(1);
    }

    // 情報
    if(policy_greedy_cnt > 0) {
        cerr << boost::format("PolicyGreedy %d times. Error %d") % policy_greedy_cnt % (policy_err_sum) << endl;
    }
    for(const auto &p : act_cnt) {
        int color_num = p.first;
        int call = color_cnt[color_num];
        int total = p.second;
        double avg = (double)total / (double)call;
        cerr << boost::format("color num: %d, call: %d, total: %d, avg: %f") % color_num % call % total % avg << endl;
    }
    cerr << boost::format("K: %d, T:%d, D:%d") % input.K % input.T % input.D << endl;
    cerr << boost::format("score: %d, elapsed: %f, turn: %d/%d") % get<2>(state.get_score()) % time_keeper.getElapsedTime() % state.turn % input.T << endl;

    // output
    Output output = Output{init_wall, state.actions};
    print_output(output);
}

int main() {
    solve();
    return 0;
}
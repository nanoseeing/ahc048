
#include <bits/stdc++.h>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>

using EigenColor = Eigen::Vector3d;

namespace Eigen {
namespace internal {

template <typename MatrixQR, typename HCoeffs, typename VectorQR>
void householder_qr_inplace_update(MatrixQR &mat, HCoeffs &hCoeffs, const VectorQR &newColumn, typename MatrixQR::Index k,
                                   typename MatrixQR::Scalar *tempData) {
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

class ColorMixer {
  public:
    struct Result {
        double squared_error;   // 真の二乗誤差
        vector<int> indices;    // 部分集合の絵の具インデックス
        vector<double> weights; // 各絵の具の重み (合計 = 1)
        EigenColor mixed_color; // 混合後の色ベクトル
    };

    // paints_input: 任意本数の絵の具ベクトル (各要素は CMY の 3 次元)
    ColorMixer(const vector<EigenColor> &paints_input) : paints(paints_input) {
        K = static_cast<int>(paints.size());
        prepare_subsets();
    }

    // 目標色 t に対し、サイズ 2～4 の部分集合で NNLS を解き、誤差が小さい上位 topN 件を返す
    vector<Result> find_topN(const EigenColor &t, int topN) const {
        const double t_norm2 = t.squaredNorm();

        struct HeapItem {
            double err;
            int subset_idx;
            bool operator<(HeapItem const &o) const {
                return err < o.err;
            }
        };
        priority_queue<HeapItem> heap;

        int total = static_cast<int>(subsets.size());

        for(int si = 0; si < total; si++) {
            const SubsetInfo &info = subsets[si];
            int n = static_cast<int>(info.indices.size()); // 2,3,4 のいずれか

            // (1) 部分集合に対応する 3×n 行列 A_S を作る
            Eigen::Matrix<double, 3, Eigen::Dynamic> A(3, n);
            for(int j = 0; j < n; j++) {
                A.col(j) = paints[info.indices[j]];
            }

            // (2) NNLS を構築して解く
            Eigen::NNLS<Eigen::Matrix<double, 3, Eigen::Dynamic>> nnls_solver;
            nnls_solver.compute(A);
            nnls_solver.setTolerance(1e-12);
            nnls_solver.setMaxIterations(2 * n);
            Eigen::VectorXd w = nnls_solver.solve(t);
            if(nnls_solver.info() != Eigen::ComputationInfo::Success) {
                // 非負制約解が収束しなかった場合は一様分配しておく
                double uni = 1.0 / n;
                w = Eigen::VectorXd::Constant(n, uni);
            }

            // (3) w >= 0 かつ sum = 1 に正規化
            double sumw = w.sum();
            if(sumw <= 1e-12) {
                double uni = 1.0 / n;
                for(int i = 0; i < n; i++)
                    w(i) = uni;
                sumw = 1.0;
            } else {
                w /= sumw;
            }

            // (4) 混合後の色 c_hat = A * w
            EigenColor c_hat = A * w;

            // (5) 真の二乗誤差 E = || t - c_hat ||^2
            double E = (t - c_hat).squaredNorm();
            double true_err = 1e4 * E; // 1e4 を掛けたスコアを使う

            // (6) ヒープに突っ込む (topN を維持)
            if(static_cast<int>(heap.size()) < topN) {
                heap.push({true_err, si});
            } else if(true_err < heap.top().err) {
                heap.pop();
                heap.push({true_err, si});
            }
        }

        // ヒープに残った topN 件を vector<Result> にまとめて返す
        int M = static_cast<int>(heap.size());
        vector<Result> results;
        results.reserve(M);

        while(!heap.empty()) {
            auto it = heap.top();
            heap.pop();
            const SubsetInfo &info = subsets[it.subset_idx];
            int n = static_cast<int>(info.indices.size());

            // A_S を再構築して NNLS をもう一度解く (重みを取り直す)
            Eigen::Matrix<double, 3, Eigen::Dynamic> A(3, n);
            for(int j = 0; j < n; j++) {
                A.col(j) = paints[info.indices[j]];
            }
            Eigen::NNLS<Eigen::Matrix<double, 3, Eigen::Dynamic>> nnls_solver;
            nnls_solver.compute(A);
            nnls_solver.setTolerance(1e-12);
            nnls_solver.setMaxIterations(2 * n);
            Eigen::VectorXd w = nnls_solver.solve(t);

            double sumw = w.sum();
            if(sumw <= 1e-12) {
                double uni = 1.0 / n;
                w = Eigen::VectorXd::Constant(n, uni);
            } else {
                w /= sumw;
            }

            EigenColor c_hat = A * w;

            Result r;
            r.squared_error = it.err;
            r.indices = info.indices;
            r.weights.resize(n);
            for(int i = 0; i < n; i++) {
                r.weights[i] = w(i);
            }
            r.mixed_color = c_hat;
            results.push_back(move(r));
        }

        sort(results.begin(), results.end(), [](auto const &a, auto const &b) { return a.squared_error < b.squared_error; });
        return results;
    }

  private:
    vector<EigenColor> paints;
    int K;

    struct SubsetInfo {
        vector<int> indices; // 部分集合の絵の具インデックス
    };
    vector<SubsetInfo> subsets;

    void prepare_subsets() {
        const vector<int> COMB_SIZES = {2, 3, 4};
        for(int sz : COMB_SIZES) {
            vector<int> comb(sz);
            function<void(int, int)> dfs = [&](int start, int depth) {
                if(depth == sz) {
                    SubsetInfo info;
                    info.indices = comb;
                    subsets.push_back(move(info));
                    return;
                }
                for(int x = start; x < K; x++) {
                    comb[depth] = x;
                    dfs(x + 1, depth + 1);
                }
            };
            dfs(0, 0);
        }
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // -------------------------------------------------------------
    // サンプル：4 本の絵の具を用意し、ColorMixer を作成
    // -------------------------------------------------------------
    vector<EigenColor> paints = {EigenColor(0.10, 0.20, 0.30), EigenColor(0.80, 0.10, 0.10), EigenColor(0.25, 0.75, 0.25), EigenColor(0.60, 0.20, 0.20)};

    ColorMixer mixer(paints);

    // 目標色 t を指定
    EigenColor t;
    t << 0.33, 0.47, 0.20;

    // 上位 5 組み合わせを取得
    int topN = 5;
    auto results = mixer.find_topN(t, topN);

    cout << fixed << setprecision(6);
    cout << "Top " << topN << " combinations:\n";
    for(int i = 0; i < (int)results.size(); i++) {
        const auto &r = results[i];
        cout << "#" << (i + 1) << "  squared_error=" << r.squared_error << "  [";
        for(int j = 0; j < (int)r.indices.size(); j++) {
            cout << r.indices[j] << (j + 1 < (int)r.indices.size() ? ", " : "");
        }
        cout << "]\n";
        cout << "    weights: [";
        for(int j = 0; j < (int)r.weights.size(); j++) {
            cout << r.weights[j] << (j + 1 < (int)r.weights.size() ? ", " : "");
        }
        cout << "]\n";
        cout << "    mixed_color: (" << r.mixed_color[0] << ", " << r.mixed_color[1] << ", " << r.mixed_color[2] << ")\n";
    }

    return 0;
}

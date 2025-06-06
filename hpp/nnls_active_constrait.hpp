#ifndef NNLS_ACTIVE_CONSTRAIT_HPP
#define NNLS_ACTIVE_CONSTRAIT_HPP

#include <Eigen/Core>
#include <Eigen/QR>
#include <algorithm>
#include <cassert>
#include <vector>

namespace Eigen {

/*
 * BVLS_PGD_SumOne: Box‐Constrained Least Squares with ∑v_i = 1
 * via Projected Gradient Descent.
 *
 *  問題:  min_v ||A v − b||²₂   s.t.  0 ≤ v_i ≤ u_i,  ∑_i v_i = 1
 *
 *  アルゴリズム:
 *   1. 勾配降下ステップ:   y = v - α * (2 Aᵀ(A v - b))
 *   2. 射影:  v_next = proj_{0 ≤ v ≤ u, ∑v=1}(y)
 *
 *  「射影」は二分探索を用いて、min‖x − y‖²  s.t. ∑x = 1,  0 ≤ x_i ≤ u_i
 *  を解く関数 projectOntoBoxWithSumOne() を実装しています。
 */

template <class MatrixType_>
class BVLS_PGD_SumOne {
  public:
    typedef MatrixType_ MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> VectorType;
    typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> RhsVectorType;

    BVLS_PGD_SumOne() = default;

    /** コンストラクタ: A, 上限ベクトル u, 反復回数, 学習率, 許容誤差 */
    BVLS_PGD_SumOne(const MatrixType &A, const std::vector<Scalar> &u, Index max_iter = 1000, Scalar step_size = Scalar(1e-3), Scalar tol = Scalar(1e-8))
        : A_(A), K_(A.cols()), max_iter_(max_iter), alpha_(step_size), tolerance_(tol) {
        assert((Index)u.size() == K_ && "u.size() must equal A.cols()");
        upperBounds_ = u;
    }

    /** A, u を再設定する */
    template <typename MatrixDerived>
    BVLS_PGD_SumOne<MatrixType> &compute(const EigenBase<MatrixDerived> &A, const std::vector<Scalar> &u) {
        A_ = A.derived();
        K_ = A_.cols();
        assert((Index)u.size() == K_ && "u.size() must equal A.cols()");
        upperBounds_ = u;
        return *this;
    }

    /** b に対する解を返す */
    VectorType solve(const RhsVectorType &b) {
        assert((Index)b.size() == A_.rows() && "b.size() must match A.rows()");
        const Index n = K_;
        // 初期解: 均等分配 (各 v_i = 1/n)、ただし上限を超えたら削る (クリップ後に再均等化)
        VectorType v = VectorType::Constant(n, Scalar(1) / Scalar(n));
        for(Index i = 0; i < n; ++i) {
            if(v(i) > upperBounds_[i]) v(i) = upperBounds_[i];
        }
        // クリップ後、∑v からずれるので再度 ∑=1 となるように単純に残りを均等分する:
        v = projectOntoBoxWithSumOne(v);

        // 反復ループ
        for(Index iter = 0; iter < max_iter_; ++iter) {
            // 1) 勾配を計算: g = 2 A^T (A v - b)
            VectorType g = A_.transpose() * (A_ * v - b);
            g *= Scalar(2.0);

            // 2) 勾配降下ステップ
            VectorType y = v - alpha_ * g;

            // 3) 射影: ∑v_i = 1, 0 ≤ v_i ≤ u_i
            VectorType v_next = projectOntoBoxWithSumOne(y);

            // 4) 収束判定
            if((v_next - v).norm() < tolerance_) {
                v = v_next;
                break;
            }
            v = v_next;
        }
        return v;
    }

    /** 学習率 (ステップサイズ) 設定 */
    BVLS_PGD_SumOne<MatrixType> &setStepSize(Scalar a) {
        alpha_ = a;
        return *this;
    }
    /** 最大反復回数 設定 */
    BVLS_PGD_SumOne<MatrixType> &setMaxIterations(Index m) {
        max_iter_ = m;
        return *this;
    }
    /** 収束許容誤差 設定 */
    BVLS_PGD_SumOne<MatrixType> &setTolerance(Scalar tol) {
        tolerance_ = tol;
        return *this;
    }

  private:
    MatrixType A_;
    Index K_ = 0;
    std::vector<Scalar> upperBounds_;
    Index max_iter_ = 1000;
    Scalar alpha_ = Scalar(1e-3);
    Scalar tolerance_ = Scalar(1e-8);

    /**
     * y を受け取って、min||x - y||²  s.t.  ∑ x_i = 1,  0 ≤ x_i ≤ u_i
     * を満たす x を返す。
     * 二分探索によりラグランジュ乗数 λ を求める。
     */
    VectorType projectOntoBoxWithSumOne(const VectorType &y) const {
        const Index n = K_;
        VectorType x(n);
        // まず λ の探索範囲:
        //  λ_low = min(y_i - u_i),  λ_high = max(y_i)
        Scalar lambda_low = std::numeric_limits<Scalar>::infinity();
        Scalar lambda_high = -std::numeric_limits<Scalar>::infinity();
        for(Index i = 0; i < n; ++i) {
            // y_i - u_i の最小値
            Scalar v1 = y(i) - upperBounds_[i];
            if(v1 < lambda_low) lambda_low = v1;
            // y_i の最大値
            if(y(i) > lambda_high) lambda_high = y(i);
        }

        // ∑ clamp(y_i - λ, 0, u_i) は λ が上がるほど減少する単調関数
        auto sumProj = [&](Scalar lambda) {
            Scalar sum = 0;
            for(Index i = 0; i < n; ++i) {
                Scalar t = y(i) - lambda;
                if(t < Scalar(0))
                    t = Scalar(0);
                else if(t > upperBounds_[i])
                    t = upperBounds_[i];
                sum += t;
            }
            return sum;
        };

        // 二分探索で λ を調整して ∑ x_i ≈ 1 にする
        Scalar target = Scalar(1);
        for(int iter = 0; iter < 50; ++iter) {
            Scalar lambda_mid = (lambda_low + lambda_high) / Scalar(2);
            Scalar s = sumProj(lambda_mid);
            if(s > target) {
                // ∑が大きい → λ を少し上げて clamp を厳しく
                lambda_low = lambda_mid;
            } else {
                lambda_high = lambda_mid;
            }
        }
        Scalar lambda = (lambda_low + lambda_high) / Scalar(2);

        // λ が決まれば、最終的に x_i = clamp(y_i - λ, 0, u_i)
        for(Index i = 0; i < n; ++i) {
            Scalar t = y(i) - lambda;
            if(t < Scalar(0))
                t = Scalar(0);
            else if(t > upperBounds_[i])
                t = upperBounds_[i];
            x(i) = t;
        }
        return x;
    }
};

} // namespace Eigen

#endif // NNLS_ACTIVE_CONSTRAIT_HPP

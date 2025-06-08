#pragma once

#include <Eigen/Core>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

/**
 * Lawson-Hanson型 アクティブセット法
 * min_x ||Ax-b||^2   s.t.  0 <= x <= u,  sum(x)=1
 * A: (m x n), b: (m), u: (n)
 */
template <class MatrixType_>
class BoxNNLS_SumToOne {
  public:
    using MatrixType = MatrixType_;
    using Scalar = typename MatrixType::Scalar;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    using Index = typename MatrixType::Index;

    BoxNNLS_SumToOne(const MatrixType& A, const std::vector<Scalar>& u, int max_iter = 1000, Scalar tol = 1e-10)
        : A_(A), u_(u), m_(A.rows()), n_(A.cols()), max_iter_(max_iter), tol_(tol) {
        assert((int)u.size() == n_);
    }

    Vector solve(const Vector& b) {
        assert(b.size() == m_);
        // 0 <= x <= u, sum(x)=1
        Vector x = Vector::Constant(n_, Scalar(1.0) / n_); // 初期値: 均等分配
        for(int i = 0; i < n_; ++i)
            x(i) = std::min(std::max(x(i), Scalar(0)), u_[i]);

        enum State { FREE, AT_ZERO, AT_UPPER };
        std::vector<State> varState(n_, FREE);

        // 初期アクティブ化
        for(int i = 0; i < n_; ++i) {
            if(x(i) <= 0) {
                x(i) = 0;
                varState[i] = AT_ZERO;
            } else if(x(i) >= u_[i]) {
                x(i) = u_[i];
                varState[i] = AT_UPPER;
            }
        }

        for(int iter = 0; iter < max_iter_; ++iter) {
            // 非アクティブ（FREE）変数だけを選ぶ
            std::vector<int> freeIdx;
            Scalar sum_act = 0;
            for(int i = 0; i < n_; ++i) {
                if(varState[i] == FREE)
                    freeIdx.push_back(i);
                else
                    sum_act += x(i);
            }
            int k = freeIdx.size();
            if(k == 0) break; // 全変数がアクティブ

            // サブシステム作成
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Af(m_, k);
            for(int i = 0; i < k; ++i)
                Af.col(i) = A_.col(freeIdx[i]);

            // KKT構築
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> KKT(k + 1, k + 1);
            KKT.setZero();
            KKT.block(0, 0, k, k) = Af.transpose() * Af;
            KKT.block(0, k, k, 1).setOnes();
            KKT.block(k, 0, 1, k).setOnes();
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rhs(k + 1);
            rhs.head(k) = Af.transpose() * b;
            rhs(k) = 1.0 - sum_act;

            // 解く
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> sol = KKT.fullPivLu().solve(rhs);
            // 新しい解をセット
            Vector x_new = x;
            for(int i = 0; i < k; ++i)
                x_new(freeIdx[i]) = sol(i);

            // 非負・上限制約違反の検出とアクティブ化
            Scalar alpha = 1.0;
            bool to_project = false;
            for(int i = 0; i < k; ++i) {
                int idx = freeIdx[i];
                if(x_new(idx) < -tol_) {
                    alpha = std::min(alpha, x(idx) / (x(idx) - x_new(idx)));
                    to_project = true;
                } else if(x_new(idx) > u_[idx] + tol_) {
                    alpha = std::min(alpha, (u_[idx] - x(idx)) / (x_new(idx) - x(idx)));
                    to_project = true;
                }
            }
            if(to_project && alpha < 1.0) {
                // 補間して張り付き点へ
                x_new = x + alpha * (x_new - x);
                for(int i = 0; i < k; ++i) {
                    int idx = freeIdx[i];
                    if(x_new(idx) < 0) x_new(idx) = 0;
                    if(x_new(idx) > u_[idx]) x_new(idx) = u_[idx];
                }
            }

            // 収束判定
            Scalar max_diff = (x - x_new).cwiseAbs().maxCoeff();
            x = x_new;
            if(max_diff < tol_) break;

            // アクティブセットの更新
            for(int i = 0; i < n_; ++i) {
                if(varState[i] == FREE) {
                    if(x(i) <= 0) {
                        x(i) = 0;
                        varState[i] = AT_ZERO;
                    } else if(x(i) >= u_[i]) {
                        x(i) = u_[i];
                        varState[i] = AT_UPPER;
                    }
                }
            }
            // (Optionally) Karush-Kuhn-Tucker条件を見てinactive化もできる（詳細は省略）
        }
        // 最終調整
        Scalar s = x.sum();
        if(std::abs(s - 1.0) > 1e-8) x /= s;
        for(int i = 0; i < n_; ++i)
            x(i) = std::min(u_[i], std::max(Scalar(0), x(i)));
        return x;
    }

  private:
    MatrixType A_;
    std::vector<Scalar> u_;
    int m_, n_;
    int max_iter_;
    Scalar tol_;
};

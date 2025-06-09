#pragma once

#include <Eigen/Core>
#include <Eigen/LU>
#include <cassert>
#include <vector>

class BVLS_BoxSum {
  public:
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    using Index = Eigen::Index;

    /// @param A         : m×n 行列
    /// @param b         : m ベクトル
    /// @param u         : 各変数の上限 uᵢ (長さ n)
    /// @param sumLower  : 合計の下限 a
    /// @param sumUpper  : 合計の上限 b
    BVLS_BoxSum(const Matrix& A, const Vector& b, const Vector& u, double sumLower, double sumUpper)
        : A_(A), b_(b), u_(u), m_(A.rows()), n_(A.cols()), sumLower_(sumLower), sumUpper_(sumUpper), state_(n_, 0), v_(Vector::Zero(n_)), sumState_(FREE) {
        assert(u_.size() == n_);
        // 初期値は下限 a を均等分配
        v_.setConstant(sumLower_ / n_);
    }

    /// 非負・箱・合計スラブ制約付き最小二乗を解く
    Vector solve() {
        bool changed = true;
        while(changed) {
            changed = false;

            // --- (1) 自由変数 F の収集と b_eff の計算
            std::vector<Index> F;
            Vector b_eff = b_;
            double usedSumU = 0.0;
            for(Index i = 0; i < n_; ++i) {
                if(state_[i] == 0) {
                    F.push_back(i);
                } else {
                    // 上限に張り付きならその分を b_eff, usedSumU から引く
                    if(state_[i] == 2) {
                        b_eff.noalias() -= A_.col(i) * u_[i];
                        usedSumU += u_[i];
                    }
                }
            }

            const Index p = static_cast<Index>(F.size());
            // 行列 A_F を組み立て
            Eigen::MatrixXd AF(m_, p);
            for(Index j = 0; j < p; ++j) {
                AF.col(j) = A_.col(F[j]);
            }

            if(sumState_ == FREE) {
                // --- (2a) 箱制約のみ: G vF = g
                Eigen::MatrixXd G = AF.transpose() * AF;
                Vector g = AF.transpose() * b_eff;
                Vector vF = G.fullPivLu().solve(g);

                // --- (3a) 変数境界チェック
                for(Index j = 0; j < p; ++j) {
                    Index i = F[j];
                    if(vF[j] < 0) {
                        state_[i] = 1; // 下限 0 に固定
                        changed = true;
                    } else if(vF[j] > u_[i]) {
                        state_[i] = 2; // 上限 uᵢ に固定
                        changed = true;
                    }
                }
                // --- (4a) 変数更新 & 合計スラブチェック
                if(!changed) {
                    for(Index j = 0; j < p; ++j) {
                        v_[F[j]] = vF[j];
                    }
                    double s = v_.sum();
                    if(s < sumLower_) {
                        sumState_ = AT_LOWER;
                        changed = true;
                    } else if(s > sumUpper_) {
                        sumState_ = AT_UPPER;
                        changed = true;
                    }
                }
            } else {
                // --- (2b) 箱制約＋合計イコール拘束
                // H = [G  1; 1ᵀ 0], rhs = [AFᵀ b_eff; targetSum − usedSumU]
                Eigen::MatrixXd G = AF.transpose() * AF;
                Eigen::MatrixXd H(p + 1, p + 1);
                H.topLeftCorner(p, p) = G;
                H.block(0, p, p, 1).setOnes();
                H.block(p, 0, 1, p).setOnes();
                H(p, p) = 0;

                Vector rhs(p + 1);
                rhs.head(p) = AF.transpose() * b_eff;
                double target = (sumState_ == AT_LOWER ? sumLower_ : sumUpper_);
                rhs[p] = target - usedSumU;

                Vector sol = H.fullPivLu().solve(rhs);
                Vector vF = sol.head(p);

                // --- (3b) 変数境界チェック
                for(Index j = 0; j < p; ++j) {
                    Index i = F[j];
                    if(vF[j] < 0) {
                        state_[i] = 1;
                        changed = true;
                    } else if(vF[j] > u_[i]) {
                        state_[i] = 2;
                        changed = true;
                    }
                }
                // --- (4b) 変数更新 & 拘束解放チェック
                if(!changed) {
                    for(Index j = 0; j < p; ++j) {
                        v_[F[j]] = vF[j];
                    }
                    double s = v_.sum();
                    if(sumState_ == AT_LOWER && s >= sumLower_ && s <= sumUpper_) {
                        sumState_ = FREE;
                        changed = true;
                    }
                    if(sumState_ == AT_UPPER && s >= sumLower_ && s <= sumUpper_) {
                        sumState_ = FREE;
                        changed = true;
                    }
                }
            }
        } // end while

        // 最後に固定変数を v_ にセット
        for(Index i = 0; i < n_; ++i) {
            if(state_[i] == 1)
                v_[i] = 0.0;
            else if(state_[i] == 2)
                v_[i] = u_[i];
        }
        return v_;
    }

  private:
    Matrix A_;
    Vector b_, u_, v_;
    Index m_, n_;

    enum SumState { FREE, AT_LOWER, AT_UPPER };
    SumState sumState_;
    double sumLower_, sumUpper_;
    std::vector<int> state_; // 0=自由, 1=下限(0), 2=上限(uᵢ)
};

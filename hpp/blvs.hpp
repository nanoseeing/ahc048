
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>
#include <limits>
#include <vector>

class BVLS_BoxSum {
  public:
    static constexpr double obj_sum = 1.0 - 1e-6; // 合計1制約の目標値
    using Matrix = Eigen::MatrixXd;
    using Vector = Eigen::VectorXd;
    using Index = Eigen::Index;

    BVLS_BoxSum(const Matrix& A, const Vector& b, const Vector& u) : A_(A), b_(b), u_(u), m_(A.rows()), n_(A.cols()) {
        assert(u_.size() == n_);
    }

    Vector solve() {
        Vector v = Vector::Zero(n_);
        std::vector<int> state(n_, 0);
        //  state[i]==0: 自由 (F)
        //            1: 下限 0 固定 (Z)
        //            2: 上限 u_i 固定 (U)
        // 初期: 全部自由にして、最後に projectToSum1 しても OK
        v.setConstant(obj_sum / n_);
        projectSum1(v);

        bool changed = true;
        while(changed) {
            changed = false;
            // 集合 F, Z, U を収集
            std::vector<Index> F;
            Vector b_eff = b_;
            for(Index i = 0; i < n_; ++i) {
                if(state[i] == 0)
                    F.push_back(i);
                else if(state[i] == 2)
                    b_eff -= A_.col(i) * u_[i];
            }

            // KKT 行列を組む
            Index p = (Index)F.size();
            Eigen::MatrixXd H(p + 1, p + 1);
            Eigen::VectorXd rhs(p + 1);
            // H = [A_F^T A_F   1; 1^T 0], rhs = [A_F^T b_eff; 1 - sum_U u]
            Eigen::MatrixXd AF(m_, p);
            for(Index j = 0; j < p; ++j)
                AF.col(j) = A_.col(F[j]);
            Eigen::MatrixXd G = AF.transpose() * AF;
            H.topLeftCorner(p, p) = G;
            H.block(0, p, p, 1).setOnes();
            H.block(p, 0, 1, p).setOnes();
            H(p, p) = 0;

            Eigen::VectorXd g = AF.transpose() * b_eff;
            rhs.head(p) = g;
            double sumU = 0;
            for(Index i = 0; i < n_; ++i)
                if(state[i] == 2) sumU += u_[i];
            rhs[p] = obj_sum - sumU;

            // KKT 系を解く
            Eigen::VectorXd sol = H.fullPivLu().solve(rhs);
            Vector vF = sol.head(p);

            // 違反チェック
            for(Index j = 0; j < p; ++j) {
                Index i = F[j];
                if(vF[j] < 0) {
                    state[i] = 1;
                    changed = true;
                } else if(vF[j] > u_[i]) {
                    state[i] = 2;
                    changed = true;
                }
            }
            // 違反がなければ自由変数を更新
            if(!changed) {
                for(Index j = 0; j < p; ++j)
                    v[F[j]] = vF[j];
            }
        } // while

        // 最後に固定組を代入
        for(Index i = 0; i < n_; ++i) {
            if(state[i] == 1)
                v[i] = 0;
            else if(state[i] == 2)
                v[i] = u_[i];
        }
        return v;
    }

  private:
    Matrix A_;
    Vector b_, u_;
    Index m_, n_;

    /// 単に合計１になるよう自由変数をシフト／スケーリング
    void projectSum1(Vector& v) {
        double s = v.sum();
        if(s > 0) {
            v /= s;
            v *= obj_sum; // 合計1.0 - 1e-6にする
        }
    }
};

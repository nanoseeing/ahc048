#ifndef _EIGEN_QP_DYNAMIC_HPP_
#define _EIGEN_QP_DYNAMIC_HPP_

#include <Eigen/Dense>

namespace EigenQP {

// -------------------------------------
// (1) defTol の宣言と特殊化
// -------------------------------------
template <typename T>
T defTol();

// double 用
template <>
inline double defTol<double>() {
    return 1E-9;
}

// float 用
template <>
inline float defTol<float>() {
    return 1E-4f;
}

// -------------------------------------
// (2) 等式制約付き QP ソルバ (動的サイズ)
// -------------------------------------
template <typename Scalar>
class QPEqSolver {
  private:
    int n; // 変数次元
    int m; // 等式制約数

    // KKT を解くためのバッファ
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Z; // (n+m)×(n+m)
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> C;              // (n+m)×1

  public:
    // コンストラクタ：n_vars=変数次元, n_const=等式制約数
    QPEqSolver(int n_vars, int n_const)
        : n(n_vars),
          m(n_const),
          Z(Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_vars + n_const, n_vars + n_const)),
          C(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(n_vars + n_const)) {
        // 下部ブロックはゼロ行列で OK（ゼロ初期化済み）
    }

    // solve:
    //   min 0.5 xᵀ Q x + cᵀ x
    //   subj to A x = b
    //
    // Q_in: (n×n), c_in: (n×1), A_in: (m×n), b_in: (m×1)
    // x_out: (n×1) に結果を返す
    void solve(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q_in, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &c_in,
               const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &A_in, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &b_in,
               Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x_out) {
        // KKT 行列 Z, 右辺 C を構築
        // Z = [ Q   Aᵀ ]
        //     [ A   0  ]
        Z.block(0, 0, n, n) = Q_in;
        Z.block(0, n, n, m) = A_in.adjoint();
        Z.block(n, 0, m, n) = A_in;
        Z.block(n, n, m, m).setZero();

        // C = [ -c ]
        //     [  b ]
        C.head(n) = -c_in;
        C.tail(m) = b_in;

        // LDLT で KKT 系を解く
        Eigen::LDLT<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>> ldlt;
        ldlt.compute(Z);
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> sol = ldlt.solve(C);

        // x_out に最初の n 要素をコピー
        x_out = sol.head(n);
    }
};

// -------------------------------------
// (3) 不等式制約付き QP ソルバ (動的サイズ)
//      内点法 / Predictor-Corrector
// -------------------------------------
template <typename Scalar>
class QPIneqSolver {
  private:
    int n; // 変数次元
    int m; // 不等式制約数

    // ワーク用ベクトル (すべてサイズは動的)
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> s; // (m×1) Slack
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> z; // (m×1) Dual (λ)

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rd; // (n×1) 残差: Qx + c - Aᵀ z
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rp; // (m×1) 残差: s + A x - b
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> rs; // (m×1) 残差: s ⊙ z

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dx; // (n×1) Δx
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ds; // (m×1) Δs
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dz; // (m×1) Δz

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x; // (n×1) 現在の x

  public:
    Scalar tolerance;
    int max_iters;

    // コンストラクタ：n_vars=変数次元, n_ineq=不等式制約数
    QPIneqSolver(int n_vars, int n_ineq)
        : n(n_vars),
          m(n_ineq),
          s(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Ones(n_ineq)),
          z(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Ones(n_ineq)),
          rd(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(n_vars)),
          rp(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(n_ineq)),
          rs(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(n_ineq)),
          dx(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(n_vars)),
          ds(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(n_ineq)),
          dz(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(n_ineq)),
          x(Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(n_vars)) {
        // defTol は必ず EigenQP::defTol<Scalar>() と名前空間を明示して呼び出す
        tolerance = EigenQP::defTol<Scalar>();
        max_iters = 250;
    }

    ~QPIneqSolver() = default;

    // solve:
    //   min 0.5 xᵀ Q x + cᵀ x
    //   subj to A x + s = b,  s ≥ 0
    //
    // Q: (n×n), c: (n×1), A: (m×n), b: (m×1), 結果は x_out (n×1)
    void solve(const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &Q, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &c,
               const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> &A, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &b,
               Eigen::Matrix<Scalar, Eigen::Dynamic, 1> &x_out) {
        const Scalar eta(0.95);
        const Scalar eps = tolerance;

        // (1) 初期化: s=1, z=1, x=0
        s.setOnes();
        z.setOnes();
        x.setZero();

        // (2) 初期残差 (x=0)
        rd = c - A.adjoint() * z;   // (n×1) = (n×m)*(m×1)
        rp = s + A * x - b;         // (m×1) = (m×n)*(n×1) − (m×1)
        rs = s.array() * z.array(); // (m×1)

        const Scalar ms = Scalar(1.0) / Scalar(m);
        Scalar mu = Scalar(n) * ms;

        for(int iter = 0; iter < max_iters; ++iter) {
            // (3) Ḡ = Q + Aᵀ * diag(z/s) * A を作成
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> z_div_s = (z.array() / s.array()).matrix(); // (m×1)
            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> DiagZdivS(m, m);
            DiagZdivS.setZero();
            DiagZdivS.diagonal() = z_div_s; // (m×m)

            Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Gbar = Q + A.adjoint() * (DiagZdivS * A); // (n×n)

            Eigen::LLT<decltype(Gbar)> lltOfGbar(Gbar);

            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dx_tmp(n);
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> ds_tmp(m);
            Eigen::Matrix<Scalar, Eigen::Dynamic, 1> dz_tmp(m);

            for(int phase = 0; phase < 2; ++phase) {
                // (4) tmp = (rs - z⊙rp) ÷ s
                Eigen::Matrix<Scalar, Eigen::Dynamic, 1> tmp = (rs.array() - z.array() * rp.array()) / s.array(); // (m×1)

                // dx = -Gbar.solve(rd + Aᵀ·tmp)
                dx_tmp = -lltOfGbar.solve(rd + A.adjoint() * tmp); // (n×1)

                // ds = A·dx_tmp − rp
                ds_tmp = A * dx_tmp - rp; // (m×1)

                // dz = -(rs + z⊙ds_tmp) ÷ s
                dz_tmp = (-(rs.array() + z.array() * ds_tmp.array())).matrix().cwiseQuotient(s); // (m×1)

                // α を求める
                Scalar alpha = Scalar(1.0);
                for(int j = 0; j < m; ++j) {
                    if(dz_tmp(j) < Scalar(0)) {
                        Scalar a = -z(j) / dz_tmp(j);
                        if(a > Scalar(0) && a < alpha) alpha = a;
                    }
                    if(ds_tmp(j) < Scalar(0)) {
                        Scalar a = -s(j) / ds_tmp(j);
                        if(a > Scalar(0) && a < alpha) alpha = a;
                    }
                }

                if(phase == 0) {
                    // 真値 μ_affine と σ を計算し、修正後残差を作成
                    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> s_aff = s + alpha * ds_tmp;
                    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> z_aff = z + alpha * dz_tmp;
                    Scalar mu_aff = (s_aff.dot(z_aff)) * ms;
                    Scalar sigma = mu_aff / mu;
                    sigma = sigma * sigma * sigma;

                    rs.array() += ds_tmp.array() * dz_tmp.array() - sigma * mu;
                } else {
                    // phase=1 のとき本更新量を保存
                    dx = dx_tmp;
                    ds = ds_tmp;
                    dz = dz_tmp;
                }
            }

            // (5) ステップ更新 (η 減衰)
            Scalar alpha = eta;
            x += alpha * dx;
            s += alpha * ds;
            z += alpha * dz;

            // (6) 残差更新
            rd = Q * x + c - A.adjoint() * z; // (n×1)
            rp = s + A * x - b;               // ← 修正：A·x (m×n×n×1)
            rs = s.array() * z.array();       // (m×1)

            mu = s.dot(z) * ms;

            // (7) 収束判定
            if((mu < eps) && (rd.norm() < eps) && (rs.norm() < eps)) {
                break;
            }
        }

        x_out = x;
    }

  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

} // namespace EigenQP

#endif // _EIGEN_QP_DYNAMIC_HPP_

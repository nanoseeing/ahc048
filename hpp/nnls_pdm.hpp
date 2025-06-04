

#pragma once

#include "common.hpp"
#include "utils.hpp"

//==========================================
// File: nnls_pg_bb.hpp
//   投影勾配法＋Barzilai–Borwein ステップによる NNLS ソルバ
//==========================================

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

// 単純体への射影関数（上で示したものと同一）
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

//==========================================
// nnls_simplex_pg_bb()
//   A ∈ ℝ^{m×n}, b ∈ ℝ^m を与えて、
//   min_{x ≥ 0, ∑_i x_i = 1} ½‖A x − b‖² を投影勾配＋BBステップで解く。
//   x は長さ n のベクトルで、初期 guess を与えておき、非負かつ合計=1 の解が返る。
//   tol: KKT 条件の残差許容誤差（最適性チェック用）
//   max_iter: 反復上限
//   戻り値: true=収束, false=限界反復到達
//==========================================
bool nnls_simplex_pg_bb(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, Eigen::VectorXd& x, double tol = 1e-12, int max_iter = 10000) {
    const int m = A.rows();
    const int n = A.cols();
    if(b.size() != m || x.size() != n) {
        std::cerr << "[nnls_simplex_pg_bb] サイズ不一致: A(" << m << "×" << n << "), b(" << b.size() << "), x(" << x.size() << ")\n";
        return false;
    }

    // A^T A, A^T b を前計算
    Eigen::MatrixXd ATA = A.transpose() * A; // (n×n)
    Eigen::VectorXd ATb = A.transpose() * b; // (n)

    // x を最初に単純体の重みで初期化しておく（たとえば一様分布）
    // 呼び出し側で希望する初期 x を与えても構いません。
    x = Eigen::VectorXd::Constant(n, 1.0 / n);

    Eigen::VectorXd g(n), g_prev(n), x_prev(n), s(n), y(n);

    // 初期勾配
    g = ATA * x - ATb;

    // BB 初期ステップ幅
    double alpha = 1.0 / (ATA.diagonal().array().maxCoeff() + 1e-8);

    for(int k = 0; k < max_iter; ++k) {
        x_prev = x;
        g_prev = g;

        // (1) 勾配ステップ→単純体への射影
        Eigen::VectorXd y_temp = x - alpha * g;
        x = ProjectOntoSimplex(y_temp);

        // (2) 勾配を再計算
        g = ATA * x - ATb;

        // (3) KKT 残差チェック（非負かつ ∑=1 の場合の最適性判定）
        //     「x_i > 0 のとき：勾配 g_i ≈ λ」, 「x_i = 0 のとき：勾配 g_i ≥ λ」
        //      が成立すれば最適。ここでは簡易的に「min(x_i, g_i)」で調べる。
        double kkt_res = 0.0;
        // まず λ の候補として、x_i > 0 な i の g_i を平均か何かで取ってもよい。
        // ここではシンプルに max_i |min(x_i, g_i)| を tol と比較：
        for(int i = 0; i < n; ++i) {
            double xi = x(i), gi = g(i);
            double tmp = std::min(xi, gi);
            kkt_res = std::max(kkt_res, std::abs(tmp));
        }
        if(kkt_res < tol) {
            return true; // 収束
        }

        // (4) BBステップ幅更新
        s = x - x_prev;
        y = g - g_prev;
        double sTy = s.dot(y);
        double sTs = s.dot(s);
        if(sTy > 1e-12) {
            double alpha_new = sTs / sTy;
            alpha = std::clamp(alpha_new, 1e-12, 1e12);
        } else {
            alpha = 1e-6;
        }
    }

    return false;
}

class ColorMixer {
  public:
    vector<Color> paints;
    int K;

    struct Result {
        double squared_error;
        vector<int> indices;
        vector<double> weights;
    };

    ColorMixer(const vector<Color>& paints_input) : paints(paints_input) {
        this->K = static_cast<int>(paints.size());
    }

    Result solve_nnls_for_indices(const vector<int>& indices, const Color& t_color, double eps = 1e-12, int max_iter = 1000) {
        Eigen::Vector3d t(t_color[0], t_color[1], t_color[2]);
        Eigen::VectorXd x0 = Eigen::VectorXd::Zero(K);
        Eigen::MatrixXd A;

        int n = static_cast<int>(indices.size());
        A.resize(3, n);
        for(int i : indices) {
            auto& col = paints[i];
            Eigen::Vector3d c(col[0], col[1], col[2]);
            A.col(i) = c;
        }

        bool ok = nnls_simplex_pg_bb(A, t, x0, eps, max_iter);

        auto err = (A * x0 - t).norm();
        auto w = x0.transpose();

        double sum_w = w.sum();
        assert(abs(sum_w - 1.0) < 1e-6); // 合計が 1 に正規化されていることを確認

        vector<double> weights;
        for(int i = 0; i < n; ++i) {
            weights.push_back(w(i));
        };

        return Result{err, indices, weights};
    }
};

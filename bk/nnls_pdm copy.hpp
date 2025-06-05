

#pragma once

#include "common.hpp"
#include "utils.hpp"

//==============================================================================
// 投影勾配法＋Barzilai–Borwein ステップによる NNLS ソルバ (By ChatGPT 4o mini)
//==============================================================================

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

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

// -----------------------------------------------------------------------------
// estimateMaxEigenvalue()
//   パワーイテレーションにより、ATA = Aᵀ A の最大固有値を推定する。
//   A: (m×n) 行列、powerIter: イテレーション回数（10～20程度で十分）
// -----------------------------------------------------------------------------
double estimateMaxEigenvalue(const Eigen::MatrixXd& A, int powerIter = 20) {
    const int n = A.cols();
    // Aᵀ A に対するパワー法
    Eigen::VectorXd v = Eigen::VectorXd::Random(n);
    v.normalize();
    for(int it = 0; it < powerIter; ++it) {
        // w ← (Aᵀ A) v
        Eigen::VectorXd w = A.transpose() * (A * v);
        double wnorm = w.norm();
        if(wnorm <= 0) break;
        v = w / wnorm;
    }
    // λ ≈ vᵀ (Aᵀ A) v
    Eigen::VectorXd Av = A * v;
    Eigen::VectorXd ATAv = A.transpose() * Av;
    double lambda = v.dot(ATAv);
    return lambda;
}

// -----------------------------------------------------------------------------
// nnls_projected_gradient_bb_clipped()
//   A ∈ R^{m×n}, b ∈ R^m を与えて、
//   min_{x ∈ simplex} ½ ||A x − b||^2 をクリップ付き BB ステップ幅で解く。
//   ・x は「x_i >=0, sum_i x_i = 1」を常に満たす（ProjectOntoSimplexを挟む）。
//   tol: KKT 条件残差許容値
//   max_iter: イテレーション上限
// -----------------------------------------------------------------------------
pair<bool, int> nnls_projected_gradient_bb_clipped(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                                   Eigen::VectorXd& x, // 初期 guess を与え、解がここに返る (size n)
                                                   double tol = 1e-7,  // KKT 条件の残差閾値
                                                   int max_iter = 1e3, // 最大イテレーション数
                                                   bool is_alpha_max_fixed = true) {
    const int m = A.rows();
    const int n = A.cols();
    if(b.size() != m || x.size() != n) {
        cerr << "[nnls_pg_bb] サイズ不一致: A(" << m << "×" << n << "), b(" << b.size() << "), x(" << x.size() << ")\n";
        return {false, -1};
    }

    // 1) 事前計算: ATA, ATb
    Eigen::MatrixXd ATA = A.transpose() * A; // (n×n)
    Eigen::VectorXd ATb = A.transpose() * b; // (n)

    // 2) λ_max = 最大固有値(AᵀA) を推定（パワー法）
    double lambda_max = estimateMaxEigenvalue(A, 20);
    if(lambda_max <= 0) lambda_max = 1e-3; // 念のためゼロ割回避

    // 3) α_max, α_min の設定（上限・下限をクリップする）
    //    上限: 0.8 / λ_max  (「1/λ_max の約80%」)
    //    下限: 1e-6 / λ_max
    double alpha_max;
    if(is_alpha_max_fixed) {
        alpha_max = 1e8; // alphaがでかいと発散することがあるので注意
    } else {
        alpha_max = 0.99 / lambda_max;
    }
    const double alpha_min = 1e-12 / lambda_max;

    // 4) 初期化: x >= 0 且つ sum(x)=1 にする
    //    └  もし呼び出し側が x ≥ 0, sum=1 を用意していなければ、
    //        ここで一様分布に初期化してもよい。
    x = ProjectOntoSimplex(x);

    // 5) 初期勾配 g = ATA*x - ATb
    Eigen::VectorXd g = ATA * x - ATb;

    // 6) 初期 step size: 1 / λ_max
    double alpha = 1.0 / lambda_max;

    // (反復用テンポラリ)
    Eigen::VectorXd x_prev(n), g_prev(n), s(n), y_vec(n);

    for(int iter = 0; iter < max_iter; ++iter) {
        // (a) 前回の保存
        x_prev = x;
        g_prev = g;

        // (b) 勾配ステップ
        Eigen::VectorXd x_tent = x - alpha * g;

        // (c) 単純体への射影 (sum=1, x_i>=0 を維持)
        x = ProjectOntoSimplex(x_tent);

        // (d) 勾配の再計算
        g = ATA * x - ATb;

        // (e) KKT 条件による収束判定
        //       「最適性残差 = max_i |min(x_i, g_i)|」が tol 未満なら終了
        double kkt_res = 0.0;
        for(int i = 0; i < n; ++i) {
            double xi = x[i], gi = g[i];
            // x_i > 0 なら g_i ≈ 0、x_i = 0 なら g_i ≥ 0
            double tmp = std::min(xi, gi);
            kkt_res = std::max(kkt_res, std::abs(tmp));
        }
        if(kkt_res < tol) {
            // 収束した
            // cout << "[BB-PGD] iter=" << iter << "  KKT_res=" << kkt_res << "\n";
            return {true, iter};
        }

        // (f) BBステップ幅更新
        s = x - x_prev;     // Δx
        y_vec = g - g_prev; // Δg
        double sty = s.dot(y_vec);
        double sts = s.squaredNorm();
        double alpha_bb;
        if(sty > 1e-16) {
            alpha_bb = sts / sty;
        } else {
            // 分母が非常に小さい・負になるときは小さな α_min を使っておく
            alpha_bb = alpha_min;
        }
        // (g) α をクリップ
        alpha = std::min(std::max(alpha_bb, alpha_min), alpha_max);
    }

    // max_iter に到達しても収束せず
    return {false, max_iter};
}

class ColorMixer {
  public:
    vector<Color> paints;
    int K;

    static constexpr double EPS = 1e-7;         // 許容誤差 (sum_w ≈ 1.0 ± epsに収束)
    static constexpr int MAX_ITER = 100;        // 簡易評価
    static constexpr int MAX_ITER_HEAVY = 1000; // 最大反復回数

    struct Result {
        double squared_error;
        vector<int> indices;
        vector<double> weights;

        int iter_cnt; // テスト用
    };

    ColorMixer(const vector<Color>& paints_input) : paints(paints_input) {
        this->K = static_cast<int>(paints.size());
    }

    vector<Result> solve_nnls_nCk(int n, int k, int max_comb, const Color& t_color) {
        // 10^5程度に抑えたい。
        const int HEAVY_THRESHOLD = 100;
        vector<vector<int>> comb_indices = choose_nCk(n, k, max_comb);
        const int comb_size = static_cast<int>(comb_indices.size());
        vector<Result> results;
        for(const auto& comb : comb_indices) {
            Result ret = solve_nnls_for_indices(comb, t_color, true, EPS, MAX_ITER);
            if(comb_size <= HEAVY_THRESHOLD && ret.squared_error >= 1.0e-4) {
                Result ret2 = solve_nnls_for_indices(comb, t_color, false, EPS, MAX_ITER_HEAVY);
                if(ret2.squared_error < ret.squared_error) {
                    ret = ret2;
                }
            }
            results.push_back(ret);
        }
        sort(results.begin(), results.end(), [](const Result& a, const Result& b) { return a.squared_error < b.squared_error; });
        return results;
    }

    Result solve_nnls_for_indices(const vector<int>& indices, const Color& t_color, double is_alpha_max_fixed = true, double eps = EPS,
                                  int max_iter = MAX_ITER) {
        int n = static_cast<int>(indices.size());
        Eigen::Vector3d t(t_color[0], t_color[1], t_color[2]);
        Eigen::VectorXd x0 = Eigen::VectorXd::Constant(n, 1.0 / double(n));
        Eigen::MatrixXd A;

        A.resize(3, n);
        for(int j = 0; j < n; ++j) {
            int paint_idx = indices[j];
            const auto& col = paints[paint_idx];
            Eigen::Vector3d c(col[0], col[1], col[2]);
            A.col(j) = c;
        }

        Eigen::VectorXd w;

        auto [ok, iter_cnt] = nnls_projected_gradient_bb_clipped(A, t, x0, eps, max_iter, is_alpha_max_fixed);
        auto err = (A * x0 - t).norm();
        w = x0.transpose();

        double sum_w = w.sum();
        assert(abs(sum_w - 1.0) < 1e-6); // 合計が 1 に正規化されていることを確認

        vector<double> weights;
        for(int i = 0; i < n; ++i) {
            weights.push_back(w(i));
        };

        return Result{err, indices, weights, iter_cnt};
    }
};

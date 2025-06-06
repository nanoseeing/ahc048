//==========================================
// File: nnls_pg_bb.hpp
//   投影勾配法＋Barzilai–Borwein ステップによる NNLS ソルバ
//==========================================

using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm> // std::sort, std::iota
#include <functional>
#include <iostream>
#include <limits>
#include <numeric> // std::iota
#include <vector>

#include "hpp/common.hpp"
#include "hpp/io.hpp"
#include "hpp/utils.hpp"

//----
// nnls_projected_gradient()
//   A ∈ ℝ^{m×n}, b ∈ ℝ^m を与えて、
//   min_{x≥0} ½‖Ax − b‖^2 を解く。
//   x はサイズ n のベクトルで、初期 guess を与えておき、最終的に非負最小二乗解が返る。
//   tol: KKT 条件の残差許容誤差（勾配と x との内積による最適性チェック）
//   max_iter: 反復の上限
//   戻り値: true=収束、false=収束しなかった
//
inline bool nnls_projected_gradient(const Eigen::MatrixXd &A, const Eigen::VectorXd &b, Eigen::VectorXd &x, double tol = 1e-12, int max_iter = 10000) {
    const int m = A.rows();
    const int n = A.cols();
    if(b.size() != m || x.size() != n) {
        std::cerr << "[nnls_pg_bb] サイズ不一致: A(" << m << "×" << n << "), b(" << b.size() << "), x(" << x.size() << ")\n";
        return false;
    }

    // 事前準備：A^T * A と A^T * b を構築しておけば、
    // 反復ごとの勾配計算が速くなる（m, n がそれなりに大きい場合）。
    Eigen::MatrixXd ATA = A.transpose() * A; // (n×n)
    Eigen::VectorXd ATb = A.transpose() * b; // (n)

    // 反復変数の初期化
    // x は呼び出し側で初期 guess を与えてもよいが、
    // ここでは一旦 0 ベクトルから始める実装にしてもよい。
    // ただし、呼び出し側で x >= 0 で初期化しておけば、より早く収束することがある。
    x = x.cwiseMax(0.0);

    // 勾配を保持するベクトル
    Eigen::VectorXd g(n), g_prev(n), x_prev(n), s(n), y(n);
    // 一度だけ初期勾配を評価
    // grad = ATA * x - ATb
    g = ATA * x - ATb;

    // BBステップの初期値
    double alpha = 1.0 / (ATA.diagonal().array().maxCoeff() + 1e-8);
    // 小さい正数を足しておかないとゼロ割の可能性がある

    // 反復ループ
    for(int k = 0; k < max_iter; ++k) {
        // (1) x_prev = x, g_prev = g
        x_prev = x;
        g_prev = g;

        // (2) 勾配ステップ → クリッピング
        Eigen::VectorXd y_temp = x - alpha * g;
        // クリッピング：負の成分を 0 に
        x = y_temp.cwiseMax(0.0);

        // (3) 勾配を再計算
        //    g = ATA * x - ATb
        g = ATA * x - ATb;

        // (4) KKT 条件による収束判定
        //    非負成分 i では x_i > 0 → 勾配 g_i ≈ 0
        //                x_i = 0 → 勾配 g_i ≥ 0
        //    つまり「最適性残差 = max_i |min(x_i, g_i)|」が tol 以下なら収束
        double kkt_res = 0.0;
        for(int i = 0; i < n; ++i) {
            double xi = x(i), gi = g(i);
            double tmp = std::min(xi, gi);
            kkt_res = std::max(kkt_res, std::abs(tmp));
        }
        if(kkt_res < tol) {
            return true; // 収束
        }

        // (5) Barzilai–Borwein ステップ長を更新
        //    s = x - x_prev,  y = g - g_prev
        s = x - x_prev;
        y = g - g_prev;
        double sTy = s.dot(y);
        double sTs = s.dot(s);

        if(sTy > 1e-12) {
            // BB formula:  α = (sᵀ s) / (sᵀ y)
            alpha = sTs / sTy;
            // 数値的安定化のため、極端に大きくならないようクリップ
            alpha = std::min(alpha, 1e12);
            alpha = std::max(alpha, 1e-12);
        } else {
            // sTy <= 0 のときは更新せず、前の alpha をそのまま使うか、
            // あるいは小さな固定値にリセットする。
            alpha = 1.0e-6;
        }
    }

    // max_iter に到達しても収束しなかった
    return false;
}

int main() {
    Eigen::MatrixXd A;

    Input input = parse_input();
    A.resize(3, input.K);
    for(int k = 0; k < input.K; ++k) {
        auto col = input.own[k];
        Eigen::Vector3d c(col[0], col[1], col[2]);
        A.col(k) = c;
    }

    TimeKeeper timer(10.0);
    for(int h : range(input.H)) {
        auto t_color = input.target[h];
        Eigen::Vector3d t(t_color[0], t_color[1], t_color[2]);
        Eigen::VectorXd x0 = Eigen::VectorXd::Zero(input.K);
        bool ok = nnls_projected_gradient(A, t, x0, 1e-9, 1000);
        // if(!ok) {
        //     std::cout << "NNLS: max_iter に到達しても厳密収束せず。\n";
        // }

        auto err = (A * x0 - t).norm();
        auto w = x0.transpose();
        vector<double> weights;
        for(int i = 0; i < input.K; ++i) {
            weights.push_back(w(i));
        }

        double true_err = 0.0;
        for(int j = 0; j < 3; ++j) {
            double now_c = 0.0;
            for(int i = 0; i < input.K; ++i) {
                now_c += input.own[i][j] * weights[i];
            }
            double diff = now_c - t[j];
            true_err += diff * diff;
        }
        true_err = sqrt(true_err);
        cpp_dump(true_err * 1e4, err * 1e4);
    }
    cpp_dump("Input K: ", input.K);
    cpp_dump("Time elapsed: ", timer.getElapsedTime());

    return 0;
}
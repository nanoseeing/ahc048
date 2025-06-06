#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <vector>

#include "project_simplex.hpp"

// ProjectOntoSimplex は上で定義済みとする

struct BatchPGDResult {
    // X_final: (n × M) の重み行列。列 k が色 k の最終解 (n×1 ベクトル)
    Eigen::MatrixXd X_final;
    // err[k]: 列 k の最終的な二乗誤差 || A_s x^{(k)} - t^{(k)} ||
    std::vector<double> err;
    // converged[k]: 列 k が収束 tol 以下になったか
    std::vector<bool> converged;
};

// バッチ版 PGD（1024 色もしくは任意 M 色）
//   A_s: (3 × n) のサブセット行列
//   T_all: (3 × M) の目標色行列（列 k が色 k のベクトル）
//   X_init: (n × M) の初期解行列。shape が合わなければ内部で (1/n,...,1/n) に初期化。
//   tol:   KKT 残差許容値（デフォルト 1e-7）
//   max_iter: 最大イテレーション回数（デフォルト 1000）
//   is_alpha_max_fixed: BBステップ幅の α_max を固定するか
//
// 返り値: BatchPGDResult{ X_final, err, converged }
//
BatchPGDResult batchSolveByPGD(const Eigen::Matrix<double, 3, Eigen::Dynamic>& A_s, const Eigen::Matrix<double, 3, Eigen::Dynamic>& T_all,
                               Eigen::MatrixXd X_init, double tol = 1e-7, int max_iter = 1000, bool is_alpha_max_fixed = true) {
    const int m = A_s.rows();   // =3
    const int n = A_s.cols();   // サブセットサイズ
    const int M = T_all.cols(); // 目標色の数（ここでは 1024 を想定）

    // 1) 前処理: ATA = A_s^T * A_s (n×n) を一度だけ計算
    Eigen::MatrixXd ATA = A_s.transpose() * A_s; // (n×n)

    // 2) 前処理: ATB = A_s^T * T_all (n×M) を一度だけ計算
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ATB(n, M);
    ATB = A_s.transpose() * T_all; // (n×3)*(3×M) → (n×M)

    // 3) λ_max をパワー法で推定（ATA は正定値なのでリッチな分解を使ってもよいが、
    //    とりあえず小規模 n ならパワー法でも十分）
    auto estimateMaxEigenvalue = [&](int powerIter = 20) {
        Eigen::VectorXd v = Eigen::VectorXd::Random(n);
        v.normalize();
        for(int it = 0; it < powerIter; ++it) {
            // w = ATA * v
            Eigen::VectorXd w = ATA * v;
            double wnorm = w.norm();
            if(wnorm <= 0) break;
            v = w / wnorm;
        }
        Eigen::VectorXd Av = ATA * v;
        return v.dot(Av);
    };
    double lambda_max = estimateMaxEigenvalue(20);
    if(lambda_max <= 0) lambda_max = 1e-3;

    // 4) α_min, α_max の設定
    double alpha_max = (is_alpha_max_fixed ? 1e12 : (0.99 / lambda_max));
    double alpha_min = 1e-12;

    // 5) 変数行列 X を初期化 (n×M)
    Eigen::MatrixXd X(n, M);
    if(X_init.rows() == n && X_init.cols() == M) {
        X = X_init;
    } else {
        // 各列を一様分布 (1/n,...,1/n)
        X = Eigen::MatrixXd::Constant(n, M, 1.0 / double(n));
    }
    // 各列を念のため単純体に射影しておく
    for(int k = 0; k < M; ++k) {
        X.col(k) = ProjectOntoSimplex(X.col(k));
    }

    // 6) 初期勾配 G = ATA*X - ATB  (n×M)
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> G(n, M);
    G = ATA * X - ATB;

    // 7) 初期ステップ幅 α
    double alpha = 1.0 / lambda_max;

    // 反復用テンポラリ
    Eigen::MatrixXd X_prev(n, M);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> G_prev(n, M);
    Eigen::MatrixXd S(n, M), Y(n, M);

    // 各列の収束フラグ
    std::vector<bool> converged(M, false);

    // 反復ループ
    for(int iter = 0; iter < max_iter; ++iter) {
        // (a) 各列の KKT 残差をチェックして収束ならフラグを立てる
        bool all_converged = true;
        for(int k = 0; k < M; ++k) {
            if(converged[k]) continue;
            // 列 k の残差
            double kkt_res = 0.0;
            const Eigen::VectorXd xk = X.col(k);
            const Eigen::VectorXd gk = G.col(k);
            for(int i = 0; i < n; ++i) {
                double xi = xk[i];
                double gi = gk[i];
                double tmp = std::min(xi, gi);
                kkt_res = std::max(kkt_res, std::abs(tmp));
            }
            if(kkt_res < tol) {
                converged[k] = true;
            } else {
                all_converged = false;
            }
        }
        if(all_converged) {
            break;
        }

        // (b) 収束していない列だけまとめて更新するため、一旦コピー
        X_prev = X;
        G_prev = G;

        // (c) 勾配ステップ: X_tent = X - α * G
        Eigen::MatrixXd X_tent = X - alpha * G;

        // (d) 各列を単純体に射影して X を更新
        for(int k = 0; k < M; ++k) {
            if(converged[k]) continue;
            X.col(k) = ProjectOntoSimplex(X_tent.col(k));
        }

        // (e) 勾配を再計算: G = ATA*X - ATB
        G = ATA * X - ATB;

        // (f) BBステップ幅を更新（全体 α を列ごとにまとめて計算）
        //     S = X - X_prev,  Y = G - G_prev
        S = X - X_prev;
        Y = G - G_prev;

        // sty = sum_k (S[:,k]·Y[:,k]),  sts = sum_k ||S[:,k]||^2
        double sty = 0.0, sts = 0.0;
        for(int k = 0; k < M; ++k) {
            sty += S.col(k).dot(Y.col(k));
            sts += S.col(k).squaredNorm();
        }
        double alpha_bb = (sty > 1e-16 ? (sts / sty) : alpha_min);
        alpha = std::min(std::max(alpha_bb, alpha_min), alpha_max);
    }

    // 8) 最終的な解 X_final を格納し、各列の誤差を計算
    BatchPGDResult result;
    result.X_final = X;
    result.err.resize(M);
    result.converged = converged;

    // E = A_s * X  → (3×M)
    Eigen::Matrix<double, 3, Eigen::Dynamic> E(3, M);
    E = A_s * X; // (3×n)*(n×M) → (3×M)

    // 目標との差 diff = E - T_all
    Eigen::Matrix<double, 3, Eigen::Dynamic> diff(3, M);
    diff = E - T_all; // (3×M)

    // 各列のノルムを err に詰める
    for(int k = 0; k < M; ++k) {
        result.err[k] = diff.col(k).norm();
    }

    return result;
}

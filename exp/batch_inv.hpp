#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cassert>
#include <vector>

// ProjectOntoSimplex は上で定義済みとする

struct BatchInvClipResult {
    // W_clip: (n × M) の重み行列。列 k が色 k のクリップ後最終解
    Eigen::MatrixXd W_clip;
    // err[k]: 列 k の二乗誤差 ‖A_s w^{(k)} - t^{(k)}‖
    std::vector<double> err;
};

// 逆行列 + クリップ方式で 1024 色一括解法
//   A_s   : (3 × n) のサブセット行列
//   T_all : (3 × M) の目標色行列（ここでは M=1024 想定）
//   eps   : クリップ後に sum=1 になっているかの許容誤差チェック（省略可）
//
// 戻り値: W_clip (n×M) および誤差ベクトル err[0..M-1]
BatchInvClipResult batchSolveByInvClip(const Eigen::Matrix<double, 3, Eigen::Dynamic>& A_s, const Eigen::Matrix<double, 3, Eigen::Dynamic>& T_all,
                                       double eps = 1e-9) {
    const int m = A_s.rows();   // =3
    const int n = A_s.cols();   // サブセットのサイズ
    const int M = T_all.cols(); // 目標色の数（ここでは 1024）

    // 1) 前処理: Gram = ATA = A_s^T * A_s  (n×n)
    Eigen::MatrixXd Gram = A_s.transpose() * A_s;

    // 2) Cholesky（または LDLT）分解
    //    正定値が成り立っていれば .llt()、数値的に怪しければ .ldlt() を使うとよい。
    Eigen::LDLT<Eigen::MatrixXd> ldlt_decomp(Gram);
    assert(ldlt_decomp.info() == Eigen::Success && "LDLT 分解に失敗");

    // 3) 右辺行列 B = A_s^T * T_all  (n×M)
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> B(n, M);
    B = A_s.transpose() * T_all; // (n×3)*(3×M) → (n×M)

    // 4) 生の最小二乗解 W_raw = (Gram)^{-1} * B  (n×M)
    //    ldlt_decomp.solve(B) で列をまとめて解く
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W_raw(n, M);
    W_raw = ldlt_decomp.solve(B); // (n×M)

    // 5) 非負化→sum=1 にクリップして W_clip を作る
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> W_clip(n, M);
    W_clip.setZero(n, M);

    // 各列ごとにクリップ＆正規化
    for(int k = 0; k < M; ++k) {
        // 5.1: 非負化と sum 計算
        double col_sum = 0.0;
        for(int i = 0; i < n; ++i) {
            double v = W_raw(i, k);
            if(v < 0.0) v = 0.0;
            W_clip(i, k) = v;
            col_sum += v;
        }
        // 5.2: 正規化（すべて 0 になってしまったら等分配）
        if(col_sum <= 0.0) {
            double uni = 1.0 / double(n);
            for(int i = 0; i < n; ++i) {
                W_clip(i, k) = uni;
            }
        } else {
            for(int i = 0; i < n; ++i) {
                W_clip(i, k) /= col_sum;
            }
        }
    }

    // 6) 各列の誤差 err[k] を計算: err = ‖ A_s * W_clip[:,k] - t^{(k)} ‖
    std::vector<double> err(M);
    // まず A_s * W_clip (3×n) * (n×M) → (3×M)
    Eigen::Matrix<double, 3, Eigen::Dynamic> E(3, M);
    E = A_s * W_clip; // (3×M)
    // 目標との差 diff = E - T_all
    Eigen::Matrix<double, 3, Eigen::Dynamic> diff(3, M);
    diff = E - T_all;

    // 列ごとにノルムを計算
    for(int k = 0; k < M; ++k) {
        err[k] = diff.col(k).norm();
    }

    // 返り値に詰める
    BatchInvClipResult result;
    result.W_clip = W_clip;
    result.err = std::move(err);
    return result;
}

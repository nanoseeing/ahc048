//-------------------------------------------
// File: mix_paints_qp.cpp
//-------------------------------------------
#include <Eigen/Dense>
#include <chrono>
#include <iostream>

#include "hpp/ex/eigen-qp-dynamic.hpp" // 自作の動的サイズ対応 QP ソルバ

using namespace std;
using namespace Eigen;
using namespace EigenQP;

/*
 * K = 20 個の絵の具 (M, C, Y) を混ぜて
 * 合計が 1.0 になるようにしつつ、
 * 目標色 t (3 次元) に近くなる二乗誤差最小の配分を求める
 *
 * - 変数 y ∈ ℝ^{19} とし、最後の絵の具 x₁₀₀ = 1 − ∑_{i=1..19} yᵢ
 *   と置き換えて求める。
 * - yᵢ ≥ 0,  ∑_{i=1..19} yᵢ ≤ 1 という 20 個の不等式を
 *   QPIneqSolver<Scalar> に渡す。
 */

int main() {
    // ----------------------------
    // (0) 定数
    // ----------------------------
    constexpr int K = 20;    // 総絵の具数
    constexpr int D = 3;     // 色ベクトル次元 (M,C,Y)
    constexpr int N = K - 1; // 19 （独立な y の数）
    constexpr int Mq = K;    // 20 個の不等式

    // ----------------------------
    // (1) 例としてランダムに絵の具を用意
    //     ここでは double 版とする
    // ----------------------------
    Matrix<double, D, K> paints;
    for(int i = 0; i < K; ++i) {
        // 実運用では任意の [M,C,Y] をここで代入する
        paints.col(i) = Matrix<double, D, 1>::Random();
    }

    // ----------------------------
    // (2) 目標色 t をランダムに設定
    // ----------------------------
    Vector<double, D> t = Vector<double, D>::Random();

    // ----------------------------
    // (3) 最終絵の具 p₁₀₀ を p_K とする
    // ----------------------------
    Vector<double, D> pK = paints.col(K - 1); // index K-1 が最後の列

    // ----------------------------
    // (4) 行列 B = [ p₁−p_K, p₂−p_K, …, p₁₉−p_K ] ∈ ℝ^{3×19}
    //     定数ベクトル r = t − p_K ∈ ℝ³
    // ----------------------------
    Matrix<double, D, N> B;
    for(int i = 0; i < N; ++i) {
        B.col(i) = paints.col(i) - pK;
    }
    Vector<double, D> r = t - pK;

    // ----------------------------
    // (5) Q = 2 * (Bᵀ B) ∈ ℝ^{19×19}
    //     c = −2 * (Bᵀ r) ∈ ℝ^{19}
    // ----------------------------
    Matrix<double, N, N> Q = 2.0 * (B.transpose() * B); // (19×19)
    Vector<double, N> c = -2.0 * (B.transpose() * r);   // (19×1)

    // ----------------------------
    // (6) 不等式制約 A·y ≤ b を A_in y + s = b_in 形にして与える
    //     ここで
    //      ・i=0..18 は yᵢ ≥ 0  →  -yᵢ ≤ 0
    //      ・i=19   は ∑ yᵢ ≤ 1 →   [1 1 … 1] y ≤ 1
    // ----------------------------
    Matrix<double, Mq, N> Aineq; // (20×19)
    Vector<double, Mq> bineq;    // (20×1)

    // (i=0..18) の −eᵢᵀ y ≤ 0
    Aineq.setZero();
    for(int i = 0; i < N; ++i) {
        Aineq(i, i) = -1.0; // row i: −yᵢ ≤ 0
        bineq(i) = 0.0;
    }
    // (i=19) の ∑ yᵢ ≤ 1
    for(int j = 0; j < N; ++j) {
        Aineq(N, j) = 1.0; // row 19: y₁ + y₂ + … + y₁₉ ≤ 1
    }
    bineq(N) = 1.0;

    // (7) QPIneqSolver<double> で解く
    // ‐ y_sol は「動的サイズ Vector (Dynamic×1)」として宣言する
    Eigen::Matrix<double, Eigen::Dynamic, 1> y_sol(N);
    {
        QPIneqSolver<double> solver(N, Mq);
        auto t0 = chrono::steady_clock::now();
        solver.solve(Q, c, Aineq, bineq, y_sol);
        auto t1 = chrono::steady_clock::now();
        double elapsed = chrono::duration<double>(t1 - t0).count();
        cout << "[IneqQP] elapsed: " << elapsed << " sec\n";
    }

    // ----------------------------
    // (8) 最終的な x ∈ ℝ^{20} を復元
    //     x₁..x₁₉ = y₁..y₁₉
    //     x₂₀     = 1 − ∑_{i=1..19} yᵢ
    // ----------------------------
    Vector<double, K> x_sol;
    double sum19 = y_sol.sum();
    for(int i = 0; i < N; ++i) {
        x_sol(i) = y_sol(i);
    }
    x_sol(K - 1) = 1.0 - sum19; // x₂₀

    // ----------------------------
    // (9) 混合後色と誤差を出力
    // ----------------------------
    // 混合後色  c_mix = ∑_{i=1..20} xᵢ pᵢ
    Vector<double, D> c_mix = Vector<double, D>::Zero();
    for(int i = 0; i < K; ++i) {
        c_mix += x_sol(i) * paints.col(i);
    }
    double mse = (c_mix - t).squaredNorm(); // 二乗誤差

    cout << "---- Solution x (sum=" << x_sol.sum() << ") ----\n";
    for(int i = 0; i < K; ++i) {
        printf("  x[%2d] = %8.6f\n", i + 1, x_sol(i));
    }
    cout << "Mixed color: [" << c_mix(0) << ", " << c_mix(1) << ", " << c_mix(2) << "]\n";
    cout << "Target color: [" << t(0) << ", " << t(1) << ", " << t(2) << "]\n";
    cout << "MSE = " << mse << "\n";

    return 0;
}

#include <bits/stdc++.h>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>

#include "hpp/ex/nnls.hpp"

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // --------------------------------------------------------------------------
    // (1) サンプル: 3×4 の行列 A と 3 ベクトル t を定義する
    // --------------------------------------------------------------------------
    // たとえば 4 本の “絵の具” を想定し、それぞれ CMY を 3 要素とする：
    // paint0 = (0.10, 0.20, 0.30)
    // paint1 = (0.80, 0.10, 0.10)
    // paint2 = (0.25, 0.75, 0.25)
    // paint3 = (0.60, 0.20, 0.20)
    Eigen::Matrix<double, 3, 4> A;
    A << 0.10, 0.80, 0.25, 0.60, 0.20, 0.10, 0.75, 0.20, 0.30, 0.10, 0.25, 0.20;

    // 目標色 t (3 要素)
    Eigen::Vector3d t;
    t << 0.33, 0.47, 0.20;

    // --------------------------------------------------------------------------
    // (2) NNLS ソルバーを構築し、A をセット
    // --------------------------------------------------------------------------
    Eigen::NNLS<Eigen::Matrix<double, 3, 4>> nnls_solver;
    nnls_solver.compute(A);

    // tolerance や maxIterations は必要に応じて設定できる：
    nnls_solver.setTolerance(1e-12);
    nnls_solver.setMaxIterations(100);

    // --------------------------------------------------------------------------
    // (3) NNLS を解いて x を得る
    // --------------------------------------------------------------------------
    Eigen::Vector4d x = nnls_solver.solve(t);

    // 結果の確認
    cout << fixed << setprecision(8);
    cout << "NNLS solution x (length 4):\n";
    for(int i = 0; i < 4; i++) {
        cout << "  x[" << i << "] = " << x[i] << "\n";
    }

    // --------------------------------------------------------------------------
    // (4) 混合後の色 c_hat = A * x を計算し、二乗誤差を出力
    // --------------------------------------------------------------------------
    Eigen::Vector3d c_hat = A * x;
    double sq_error = (t - c_hat).squaredNorm();

    cout << "\nMixed color c_hat = (" << c_hat[0] << ", " << c_hat[1] << ", " << c_hat[2] << ")\n";
    cout << "Squared error ||t - c_hat||^2 = " << sq_error << "\n";

    return 0;
}
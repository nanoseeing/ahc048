#include <Eigen/Core>
#include <iostream>

#include "hpp/nnls_active_constrait.hpp" // 上記ヘッダ

int main() {
    // 例：A は 3×4 行列
    Eigen::Matrix<double, 3, 4> A;
    A << 0.10, 0.80, 0.25, 0.60, 0.20, 0.10, 0.75, 0.20, 0.30, 0.10, 0.25, 0.20;

    // 上限ベクトル u (長さ 4)
    // ここを {0.5, 0.4, 1.0, 0.3} とすると、以前のアクティブセット実装で
    // エラーが出ていたケースに対応できます。
    std::vector<double> u = {1.0, 0.4, 0.45, 0.3};

    // 目標ベクトル t (長さ 3)
    Eigen::Vector3d t(0.5, 0.3, 0.2);

    // PGD ソルバを構築
    Eigen::BVLS_PGD_SumOne<Eigen::Matrix<double, 3, 4>> solver(A, u);
    // 必要に応じてパラメータを変更:
    solver.setMaxIterations(5000).setStepSize(1e-3).setTolerance(1e-8);

    // 解を求める (4×1 ベクトル)
    Eigen::Vector4d v = solver.solve(t);

    std::cout << "Solution v:\n" << v.transpose() << "\n";
    std::cout << "Residual ||A v - t||^2 = " << (A * v - t).squaredNorm() << "\n";
    return 0;
}

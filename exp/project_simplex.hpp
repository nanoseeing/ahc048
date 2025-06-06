#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <vector>

// ----------------------------------------
// ProjectOntoSimplex(v)
//   入力 v ∈ R^n を “単純体 { x | x_i ≥ 0, ∑_i x_i = 1 }” に射影して返す。
//   内部では O(n log n) のソートを行う。
// ----------------------------------------
Eigen::VectorXd ProjectOntoSimplex(const Eigen::VectorXd& v) {
    const int n = v.size();
    // ソート用にコピー
    std::vector<double> u(n);
    for(int i = 0; i < n; ++i) {
        u[i] = v[i];
    }
    std::sort(u.begin(), u.end(), std::greater<double>());

    // 累積和
    std::vector<double> cumsum(n);
    cumsum[0] = u[0];
    for(int i = 1; i < n; ++i) {
        cumsum[i] = cumsum[i - 1] + u[i];
    }

    // ρ, θ を求める
    int rho = -1;
    double theta = 0.0;
    for(int j = 0; j < n; ++j) {
        double t = (cumsum[j] - 1.0) / double(j + 1);
        if(u[j] - t > 0) {
            rho = j;
            theta = t;
        }
    }
    // もし rho < 0 なら、一様分布に落とす
    if(rho < 0) {
        return Eigen::VectorXd::Constant(n, 1.0 / double(n));
    }
    // 投影結果を返す
    Eigen::VectorXd w(n);
    for(int i = 0; i < n; ++i) {
        w[i] = std::max(v[i] - theta, 0.0);
    }
    return w;
}

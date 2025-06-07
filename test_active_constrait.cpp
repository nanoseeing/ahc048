#include <Eigen/Core>
#include <iostream>

// #include "hpp/nnls_active_constrait.hpp" // 上記ヘッダ
#include "hpp/blvs.hpp"
#include "hpp/common.hpp"
#include "hpp/io.hpp"
#include "hpp/utils.hpp"

double true_error(Input &input, vector<double> &weights, vector<int> &indices, Color &target_color) {
    double true_err = 0.0;
    for(int j = 0; j < 3; ++j) {
        double now_c = 0.0;
        for(int i = 0; i < (int)indices.size(); ++i) {
            int idx = indices[i];
            now_c += input.own[idx][j] * weights[i];
        }
        double diff = now_c - target_color[j];
        true_err += diff * diff;
    }
    return sqrt(true_err);
}

int main() {
    Input input = parse_input();
    TimeKeeper timer(100.0);

    vector<double> true_errors;
    for(int h = 0; h < input.H; ++h) {
        vector<int> best_subset;
        double best_error = 1e9;

        Eigen::MatrixXd A_ext;
        A_ext.resize(3, input.K);
        for(int k = 0; k < input.K; ++k) {
            auto col = input.own[k];
            Eigen::Vector3d c(col[0], col[1], col[2]);
            A_ext.block<3, 1>(0, k) = c; // 上3行
        }

        Eigen::Vector3d t_ext;
        t_ext(0) = input.target[h][0];
        t_ext(1) = input.target[h][1];
        t_ext(2) = input.target[h][2];

        Eigen::VectorXd u;
        u.resize(input.K);
        for(int i = 0; i < input.K; ++i) {
            u(i) = 0.5;
        }

        BVLS_BoxSum solver(A_ext, t_ext, u);
        Eigen::VectorXd x = solver.solve();
        double sum_w = x.sum();

        // assert(abs(sum_w - 1.0) < 1e-6); // 合計1制約
        for(int i = 0; i < input.K; ++i) {
            assert(x(i) >= 0.0);  // 非負制約
            assert(x(i) <= u(i)); // 上限制約
        }

        vector<double> weights(input.K);
        int nonw_zero_count = 0;
        for(int i = 0; i < input.K; ++i) {
            weights[i] = x(i) / sum_w;
            if(weights[i] > 1e-6) {
                nonw_zero_count++;
            }
        }
        double sum_new_w = accumulate(weights.begin(), weights.end(), 0.0);
        assert(abs(sum_new_w - 1.0) < 1e-6);

        vector<int> indices(input.K);
        for(int i = 0; i < input.K; ++i) {
            indices[i] = i;
        }
        double true_err = true_error(input, weights, indices, input.target[h]);

        true_errors.push_back(true_err);

        cpp_dump(h, true_err * 1e4, nonw_zero_count, abs(sum_w - 1.0));
    }

    return 0;
}

// int main() {
//     // 例：A は 3×4 行列
//     Eigen::Matrix<double, 3, 4> A;
//     A << 0.10, 0.80, 0.25, 0.60, 0.20, 0.10, 0.75, 0.20, 0.30, 0.10, 0.25, 0.20;

//     // 上限ベクトル u (長さ 4)
//     // ここを {0.5, 0.4, 1.0, 0.3} とすると、以前のアクティブセット実装で
//     // エラーが出ていたケースに対応できます。
//     std::vector<double> u = {1.0, 0.4, 0.45, 0.3};

//     // 目標ベクトル t (長さ 3)
//     Eigen::Vector3d t(0.5, 0.3, 0.2);

//     // PGD ソルバを構築
//     Eigen::BVLS_PGD_SumOne<Eigen::Matrix<double, 3, 4>> solver(A, u);
//     // 必要に応じてパラメータを変更:
//     solver.setMaxIterations(5000).setStepSize(1e-3).setTolerance(1e-8);

//     // 解を求める (4×1 ベクトル)
//     Eigen::Vector4d v = solver.solve(t);

//     std::cout << "Solution v:\n" << v.transpose() << "\n";
//     std::cout << "Residual ||A v - t||^2 = " << (A * v - t).squaredNorm() << "\n";
//     return 0;
// }

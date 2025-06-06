#include <bits/stdc++.h>
using namespace std;
#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <vector>

#include "hpp/common.hpp"
#include "hpp/ex/nnls.hpp"
#include "hpp/game.hpp"
#include "hpp/io.hpp"
#include "hpp/utils.hpp"

double true_error(Input &input, vector<double> &weights, vector<int> &indices, Color &target_color) {
    double true_err = 0.0;
    for(int j = 0; j < 3; ++j) {
        double now_c = 0.0;
        for(int i : indices) {
            now_c += input.own[i][j] * weights[i];
        }
        double diff = now_c - target_color[j];
        true_err += diff * diff;
    }
    return true_err;
}

int main() {
    Input input = parse_input();

    vector<int> indices(input.K);
    iota(indices.begin(), indices.end(), 0);

    const int K = input.K;
    Eigen::MatrixXd A_ext;
    A_ext.resize(4, K);
    for(int k = 0; k < K; ++k) {
        auto col = input.own[k];
        Eigen::Vector3d c(col[0], col[1], col[2]);
        A_ext.block<3, 1>(0, k) = c; // 上3行
    }
    A_ext.row(3).setOnes();

    TimeKeeper timer(100.0);
    vector<double> errors(input.H);

    vector<double> true_errors(input.H);
    vector<int> non_zero_nums(input.H);
    for(int h = 0; h < input.H; ++h) {
        Eigen::NNLS<Eigen::MatrixXd> nnls_solver;
        nnls_solver.compute(A_ext);
        nnls_solver.setTolerance(1e-7);
        nnls_solver.setMaxIterations(50);
        Eigen::Vector4d t_ext;
        t_ext(0) = input.target[h][0];
        t_ext(1) = input.target[h][1];
        t_ext(2) = input.target[h][2];
        t_ext(3) = 1.0; // 「和が１になる」項を擬似的に加える

        Eigen::VectorXd x = nnls_solver.solve(t_ext);
        Eigen::Vector4d c_hat_ext = A_ext * x;
        double sq_error_ext = (t_ext - c_hat_ext).squaredNorm();
        double sq_error_rgb =
            (Eigen::Vector3d(c_hat_ext(0), c_hat_ext(1), c_hat_ext(2)) - Eigen::Vector3d(input.target[h][0], input.target[h][1], input.target[h][2]))
                .squaredNorm();

        double sum_w = x.sum();
        vector<double> weights(K);
        int nonw_zero_count = 0;
        for(int i = 0; i < K; ++i) {
            weights[i] = x(i) / sum_w;
            if(weights[i] > 1e-6) {
                nonw_zero_count++;
            }
        }

        double sum_new_w = accumulate(weights.begin(), weights.end(), 0.0);

        assert(abs(sum_new_w - 1.0) < 1e-6);
        double true_err = true_error(input, weights, indices, input.target[h]);

        true_errors[h] = true_err + 10;
        non_zero_nums[h] = nonw_zero_count;
        // cerr << boost::format(" %d: ext_sq_error = %.4f, rgb_err = %.4f, sum_w = %.6f, true_err = %.4f\n") % h % (sq_error_ext * 1e4) % (sq_error_rgb * 1e4)
        // %
        //             sum_w % (true_err * 1e4);
    }

    for(auto &te : true_errors) {
        cerr << boost::format("%.4f ") % (te * 1e4);
    }
    cerr << endl;
    for(auto &td : non_zero_nums) {
        cerr << boost::format("%d ") % td;
    }
    cerr << endl;

    // cerr << "NNLS (拡張行列版) finished.\n";
    // cerr << "Total time: " << timer.getElapsedTime() << " seconds.\n";
    return 0;
}

// int main() {
//     Input input = parse_input();

//     Eigen::MatrixXd A;
//     A.resize(3, input.K);
//     for(int k = 0; k < input.K; ++k) {
//         auto col = input.own[k];
//         Eigen::Vector3d c(col[0], col[1], col[2]);
//         A.col(k) = c;
//     }

//     vector<int> indices(input.K);
//     iota(indices.begin(), indices.end(), 0);

//     TimeKeeper timer(100.0);
//     vector<double> errors(input.H);
//     for(int h : range(input.H)) {
//         Eigen::NNLS<Eigen::MatrixXd> nnls_solver;
//         nnls_solver.compute(A);

//         nnls_solver.setTolerance(1e-12);
//         nnls_solver.setMaxIterations(10000);
//         Eigen::Vector3d t(input.target[h][0], input.target[h][1], input.target[h][2]);
//         Eigen::VectorXd x = nnls_solver.solve(t);
//         Eigen::Vector3d c_hat = A * x;
//         double sq_error = (t - c_hat).squaredNorm();
//         double sum_w = x.sum();
//         vector<double> weights(input.K);
//         for(int i = 0; i < input.K; ++i) {
//             weights[i] = x(i) / sum_w; // 重みを正規化
//         }
//         double sum_new_w = accumulate(weights.begin(), weights.end(), 0.0);
//         assert(abs(sum_new_w - 1.0) < 1e-6); // 重みの合計が 1 に近いことを確認
//         double true_err = true_error(input, weights, indices, input.target[h]);
//         double diff = abs(sq_error - true_err);
//         cerr << boost::format(" %d: sq_error = %.4f, true_err = %.4f, diff = %.4f\n") % h % (sq_error * 1e4) % (true_err * 1e4) % (diff * 1e4);
//         // cerr << " " << h << ": " << sq_error * 1e4 << "\n";
//     }
//     cerr << "NNLS finished.\n";
//     cerr << "Total time: " << timer.getElapsedTime() << " seconds.\n";

//     return 0;
// }

// int main() {
//     Input input = parse_input();

//     Eigen::Matrix<double, 3, 20> A;
//     // A.resize(3, input.K);
//     for(int k = 0; k < input.K; ++k) {
//         auto col = input.own[k];
//         Eigen::Vector3d c(col[0], col[1], col[2]);
//         A.col(k) = c;
//     }

//     Eigen::NNLS<Eigen::Matrix<double, 3, 20>> nnls_solver;
//     nnls_solver.compute(A);

//     nnls_solver.setTolerance(1e-7);
//     nnls_solver.setMaxIterations(30);

//     TimeKeeper timer(100.0);
//     vector<double> errors(input.H);
//     for(int h : range(input.H)) {
//         Eigen::Vector3d t(input.target[h][0], input.target[h][1], input.target[h][2]);
//         Eigen::Vector<double, 20> x = nnls_solver.solve(t);
//         Eigen::Vector3d c_hat = A * x;
//         double sq_error = (t - c_hat).squaredNorm();
//         // cerr << " " << h << ": " << sq_error * 1e4 << "\n";
//     }
//     cerr << "NNLS finished.\n";
//     cerr << "Total time: " << timer.getElapsedTime() << " seconds.\n";

//     return 0;
// }
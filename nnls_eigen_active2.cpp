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
        for(int i = 0; i < (int)indices.size(); ++i) {
            int idx = indices[i];
            now_c += input.own[idx][j] * weights[i];
        }
        double diff = now_c - target_color[j];
        true_err += diff * diff;
    }
    return sqrt(true_err);
}

// vector<vector<int>> construct_subsets(int size, int k) {
//     vector<vector<int>> subsets;
//     vector<int> comb(size);
//     function<void(int, int)> dfs = [&](int start, int depth) {
//         if(depth == size) {
//             subsets.emplace_back(comb.begin(), comb.end());
//             return;
//         }
//         for(int x = start; x < k; x++) {
//             comb[depth] = x;
//             dfs(x + 1, depth + 1);
//         }
//     };
//     dfs(0, 0);

//     return subsets;
// }

#include "hpp/nnls.hpp"

int main() {
    Input input = parse_input();

    const int COMB_K = 3;
    auto subsets = construct_subsets(COMB_K, input.K);

    TimeKeeper timer(100.0);

    vector<double> best_errs(input.H);
    for(int h = 0; h < input.H; ++h) {
        vector<double> true_errors;
        vector<int> best_subset;
        double best_error = 1e9;
        for(auto &subset : subsets) {
            int sub_size = subset.size();

            Eigen::MatrixXd A_ext;
            A_ext.resize(3, sub_size);
            for(int k = 0; k < sub_size; ++k) {
                auto col = input.own[subset[k]];
                Eigen::Vector3d c(col[0], col[1], col[2]);
                A_ext.block<3, 1>(0, k) = c; // 上3行
            }

            Eigen::Vector3d t_ext;
            t_ext(0) = input.target[h][0];
            t_ext(1) = input.target[h][1];
            t_ext(2) = input.target[h][2];

            Eigen::VectorXd x(sub_size);
            bool ok = nnls_projected_gradient_bb_clipped(A_ext, t_ext, x, 1e-12, 100, true);
            double sum_w = x.sum();

            assert(abs(sum_w - 1.0) < 1e-6);

            vector<double> weights(sub_size);
            int nonw_zero_count = 0;
            for(int i = 0; i < sub_size; ++i) {
                weights[i] = x(i) / sum_w;
                if(weights[i] > 1e-6) {
                    nonw_zero_count++;
                }
            }

            double sum_new_w = accumulate(weights.begin(), weights.end(), 0.0);
            assert(abs(sum_new_w - 1.0) < 1e-6);
            double true_err = true_error(input, weights, subset, input.target[h]);

            true_errors.push_back(true_err);
            if(true_err < best_error) {
                best_error = true_err;
                best_subset = subset;
            }
        }
        double min_true_err = *min_element(true_errors.begin(), true_errors.end());
        best_errs[h] = min_true_err;

        cpp_dump(h, min_true_err * 1e4);
    }

    cpp_dump(input.K, subsets.size());

    for(auto &te : best_errs) {
        cout << boost::format("%.4f ") % (te * 1e4);
    }
    cout << endl;

    return 0;
}

// =======================
// アクティブセットの全サブセット + 合計1に正規化する4次元（なぜか誤差がでる。行列の条件が違うとだめらしい。）
// =======================
// int main() {
//     Input input = parse_input();

//     const int COMB_K = 3;
//     auto subsets = construct_subsets(COMB_K, input.K);

//     TimeKeeper timer(100.0);

//     vector<double> best_errs(input.H);
//     for(int h = 0; h < input.H; ++h) {
//         vector<double> true_errors;
//         vector<int> best_subset;
//         double best_error = 1e9;
//         for(auto &subset : subsets) {
//             int sub_size = subset.size();

//             Eigen::MatrixXd A_ext;
//             A_ext.resize(4, sub_size);
//             for(int k = 0; k < sub_size; ++k) {
//                 auto col = input.own[subset[k]];
//                 Eigen::Vector3d c(col[0], col[1], col[2]);
//                 A_ext.block<3, 1>(0, k) = c; // 上3行
//             }
//             A_ext.row(3).setOnes();

//             Eigen::NNLS<Eigen::MatrixXd> nnls_solver;
//             nnls_solver.compute(A_ext);
//             nnls_solver.setTolerance(1e-7);
//             nnls_solver.setMaxIterations(50);

//             Eigen::Vector4d t_ext;
//             t_ext(0) = input.target[h][0];
//             t_ext(1) = input.target[h][1];
//             t_ext(2) = input.target[h][2];
//             t_ext(3) = 1.0; // 「和が１になる」項を擬似的に加える

//             Eigen::VectorXd x = nnls_solver.solve(t_ext);
//             double sum_w = x.sum();

//             // この段階では1にならないことがある。
//             // assert(abs(sum_w - 1.0) < 1e-6);

//             vector<double> weights(sub_size);
//             int nonw_zero_count = 0;
//             for(int i = 0; i < sub_size; ++i) {
//                 weights[i] = x(i) / sum_w;
//                 if(weights[i] > 1e-6) {
//                     nonw_zero_count++;
//                 }
//             }

//             double sum_new_w = accumulate(weights.begin(), weights.end(), 0.0);
//             assert(abs(sum_new_w - 1.0) < 1e-6);
//             double true_err = true_error(input, weights, subset, input.target[h]);

//             true_errors.push_back(true_err);
//             if(true_err < best_error) {
//                 best_error = true_err;
//                 best_subset = subset;
//             }
//         }
//         double min_true_err = *min_element(true_errors.begin(), true_errors.end());
//         best_errs[h] = min_true_err;

//         // Eigen::MatrixXd A_ext;
//         // A_ext.resize(4, COMB_K);
//         // for(int k = 0; k < COMB_K; ++k) {
//         //     auto col = input.own[best_subset[k]];
//         //     Eigen::Vector3d c(col[0], col[1], col[2]);
//         //     A_ext.block<3, 1>(0, k) = c; // 上3行
//         // }
//         // A_ext.row(3).setOnes();

//         cpp_dump(h, min_true_err * 1e4);
//     }

//     cpp_dump(input.K, subsets.size());

//     for(auto &te : best_errs) {
//         cout << boost::format("%.4f ") % (te * 1e4);
//     }
//     cout << endl;

//     return 0;
// }

// =======================
// アクティブセット + 合計1になるように4次元にする（全部OK!!!）
// =======================
// int main() {
//     Input input = parse_input();

//     TimeKeeper timer(100.0);

//     vector<int> indices(input.K);
//     iota(indices.begin(), indices.end(), 0);

//     vector<double> true_errors;
//     // vector<vector<int>> non_zero_counts;
//     for(int h = 0; h < input.H; ++h) {
//         Eigen::MatrixXd A_ext;
//         A_ext.resize(4, input.K);
//         for(int k = 0; k < input.K; ++k) {
//             auto col = input.own[k];
//             Eigen::Vector3d c(col[0], col[1], col[2]);
//             A_ext.block<3, 1>(0, k) = c; // 上3行
//         }
//         A_ext.row(3).setOnes();

//         Eigen::NNLS<Eigen::MatrixXd> nnls_solver;
//         nnls_solver.compute(A_ext);
//         nnls_solver.setTolerance(1e-7);
//         nnls_solver.setMaxIterations(50);

//         Eigen::Vector4d t_ext;
//         t_ext(0) = input.target[h][0];
//         t_ext(1) = input.target[h][1];
//         t_ext(2) = input.target[h][2];
//         t_ext(3) = 1.0; // 「和が１になる」項を擬似的に加える

//         Eigen::VectorXd x = nnls_solver.solve(t_ext);
//         Eigen::Vector4d c_hat_ext = A_ext * x;
//         double sq_error_ext = (t_ext - c_hat_ext).squaredNorm();
//         double sq_error_rgb =
//             (Eigen::Vector3d(c_hat_ext(0), c_hat_ext(1), c_hat_ext(2)) - Eigen::Vector3d(input.target[h][0], input.target[h][1], input.target[h][2]))
//                 .squaredNorm();

//         double sum_w = 0.0;
//         for(int i = 0; i < input.K; ++i) {
//             sum_w += x(i);
//         }

//         assert(abs(sum_w - 1.0) < 1e-6); // 重みの合計が 1 に近いことを確認

//         vector<double> tmp_weights(input.K);
//         vector<double> weights_ok;
//         vector<int> non_zero_indices;
//         for(int i = 0; i < input.K; ++i) {
//             tmp_weights[i] = x(i) / sum_w;
//             if(tmp_weights[i] > 1e-6) {
//                 non_zero_indices.push_back(i);
//                 weights_ok.push_back(tmp_weights[i]);
//             }
//         }

//         double sum_new_w = accumulate(weights_ok.begin(), weights_ok.end(), 0.0);
//         assert(abs(sum_new_w - 1.0) < 1e-6); // 重みの合計が 1 に近いことを確認

//         double true_err = true_error(input, weights_ok, non_zero_indices, input.target[h]);
//         true_errors.push_back(true_err);

//         // 確認をする
//         Eigen::MatrixXd A2;
//         int newK = non_zero_indices.size();
//         A2.resize(4, newK);
//         for(int k = 0; k < newK; ++k) {
//             auto col = input.own[non_zero_indices[k]];
//             Eigen::Vector3d c(col[0], col[1], col[2]);
//             A2.block<3, 1>(0, k) = c; // 上3行
//         }
//         A2.row(3).setOnes();

//         Eigen::NNLS<Eigen::MatrixXd> nnls_solver2;
//         nnls_solver2.compute(A2);
//         nnls_solver2.setTolerance(1e-12);
//         nnls_solver2.setMaxIterations(10000000);

//         Eigen::VectorXd x2 = nnls_solver2.solve(t_ext);
//         double sum_w2 = 0.0;
//         for(int i = 0; i < newK; ++i) {
//             sum_w2 += x2(i);
//         }

//         vector<double> weights2(newK);
//         for(int i = 0; i < newK; ++i) {
//             weights2[i] = x2(i) / sum_w2;
//         }

//         double sum_new_w2 = accumulate(weights2.begin(), weights2.end(), 0.0);
//         assert(abs(sum_new_w2 - 1.0) < 1e-6); // 重みの合計が 1 に近いことを確認

//         double true_err2 = true_error(input, weights2, non_zero_indices, input.target[h]);

//         // SVDによる特異値分解
//         // Eigen::JacobiSVD<Eigen::MatrixXd> svd(A2);
//         // Eigen::VectorXd singular_values = svd.singularValues();
//         // double sigma_max = singular_values(0);
//         // double sigma_min = singular_values(singular_values.size() - 1);
//         // double cond = sigma_max / sigma_min;

//         cpp_dump(h, true_err * 1e4, true_err2 * 1e4, weights_ok);
//     }

//     // 統計確認
//     double sum_true_err = accumulate(true_errors.begin(), true_errors.end(), 0.0);
//     double avg_true_err = sum_true_err / input.H;
//     double std_true_err = 0.0;
//     for(double te : true_errors) {
//         std_true_err += (te - avg_true_err) * (te - avg_true_err);
//     }
//     std_true_err = sqrt(std_true_err / input.H);
//     cpp_dump(sum_true_err, avg_true_err, std_true_err);

//     // for(auto &te : true_errors) {
//     //     cout << boost::format("%.4f ") % (te * 1e4);
//     // }
//     // cout << endl;

//     return 0;
// }
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

/**
 * ProjectOntoSimplex(v)
 *   v を「x >= 0, sum_i x_i = 1」の単体(Δ) にユークリッド距離最小で射影する。
 */
Eigen::VectorXd ProjectOntoSimplex(const Eigen::VectorXd &v) {
    const int n = v.size();
    vector<double> u(n);
    for(int i = 0; i < n; ++i) {
        u[i] = v[i];
    }
    // 1) 降順ソート
    sort(u.begin(), u.end(), greater<double>());

    // 2) 累積和を計算
    vector<double> cumsum(n);
    cumsum[0] = u[0];
    for(int i = 1; i < n; ++i) {
        cumsum[i] = cumsum[i - 1] + u[i];
    }

    // 3) ρ と θ を求める
    int rho = -1;
    double theta = 0.0;
    for(int j = 0; j < n; ++j) {
        double t = (cumsum[j] - 1.0) / (j + 1);
        if(u[j] > t) {
            rho = j;
            theta = t;
        }
    }

    Eigen::VectorXd w(n);
    if(rho < 0) {
        // 全要素が1/n を返す
        return Eigen::VectorXd::Constant(n, 1.0 / n);
    }
    // 4) v - θ を最大(0)
    for(int i = 0; i < n; ++i) {
        w[i] = max(v[i] - theta, 0.0);
    }
    return w;
}

/**
 * estimateMaxEigenvalue(A, powerIter)
 *   AᵀA の最大固有値をパワーイテレーションで推定する。
 *   powerIter: 反復回数（10～20 ほどで大まかに収束する）
 */
double estimateMaxEigenvalue(const Eigen::MatrixXd &A, int powerIter = 20) {
    int n = A.cols();
    Eigen::VectorXd v = Eigen::VectorXd::Random(n);
    v.normalize();
    for(int it = 0; it < powerIter; ++it) {
        Eigen::VectorXd w = A.transpose() * (A * v); // (AᵀA) v
        double wnorm = w.norm();
        if(wnorm <= 0) break;
        v = w / wnorm;
    }
    Eigen::VectorXd Av = A * v;
    Eigen::VectorXd ATAv = A.transpose() * Av;
    double lambda = v.dot(ATAv);
    return lambda;
}

/**
 * nnls_projected_gradient_bb_clipped(A, b, x, tol, max_iter, is_alpha_max_fixed)
 *
 *   A ∈ R^{m×n}, b ∈ R^m を与えて、
 *   min_{x ∈ Δ} ½ ||A x − b||_2^2 を BB ステップ幅 + クリップ付き PGD で解く。
 *   Δ = { x | x_i ≥ 0, ∑_i x_i = 1 }
 *
 *   x: 長さ n のベクトル。呼び出し前に「x = VectorXd::Constant(n, 1.0/n)」などで初期化し、
 *      解をここへ返す。
 *   tol: KKT 残差の許容値（単体制約を考慮したもの）。1e-12～1e-14 程度を推奨。
 *   max_iter: 最大イテレーション数。1e4～2e4 程度を推奨。
 *   is_alpha_max_fixed:
 *       false → α_max = 0.99/λ_max にする（推奨）
 *       true  → α_max = 1e8 など大きな値（実質クリップなし。推奨しない）
 *
 *   戻り値: true なら tol 内に収束した。false なら max_iter 打ち切り。
 */
bool nnls_projected_gradient_bb_clipped(const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
                                        Eigen::VectorXd &x,   // 初期 guess を与え、解はここに返る (size n)
                                        double tol = 1e-14,   // KKT 条件の残差閾値
                                        int max_iter = 20000, // 最大イテレーション数
                                        bool is_alpha_max_fixed = false) {
    int m = A.rows();
    int n = A.cols();
    if(b.size() != m || x.size() != n) {
        cerr << "[nnls_pg_bb] サイズ不一致 A(" << m << "×" << n << "), b(" << b.size() << "), x(" << x.size() << ")\n";
        return false;
    }

    // 1) ATA, ATb の計算
    Eigen::MatrixXd ATA = A.transpose() * A; // n×n
    Eigen::VectorXd ATb = A.transpose() * b; // n

    // 2) λ_max を推定
    double lambda_max = estimateMaxEigenvalue(A, 50);
    if(lambda_max <= 0) lambda_max = 1e-6;

    // 3) α_max, α_min の設定
    double alpha_max;
    if(is_alpha_max_fixed) {
        alpha_max = 1e8; // どんなに大きくてもここまで、を許す（推奨しない）
    } else {
        alpha_max = 0.99 / lambda_max;
    }
    const double alpha_min = 1e-12 / lambda_max;

    // 4) x の初期化: x >= 0, sum=1 にする
    //    呼び出し側でほぼ uniform に初期化してある前提だが、念のため再射影
    x = ProjectOntoSimplex(x);

    // 5) 初期勾配 g = ATA*x - ATb
    Eigen::VectorXd g = ATA * x - ATb;

    // 6) 初期 step size
    double alpha = 1.0 / lambda_max;

    // 7) 反復用
    Eigen::VectorXd x_prev = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd g_prev = Eigen::VectorXd::Zero(n);
    Eigen::VectorXd s(n), y_vec(n);

    for(int iter = 0; iter < max_iter; ++iter) {
        // (a) 前回値保存
        x_prev = x;
        g_prev = g;

        // (b) 勾配ステップ
        Eigen::VectorXd x_tent = x - alpha * g;

        // (c) 単体に射影
        x = ProjectOntoSimplex(x_tent);

        // (d) 新しい勾配 g = ATA*x - ATb
        g = ATA * x - ATb;

        // (e) KKT 条件による収束判定（“単体用” に修正）
        //    単体 (x>=0, sum=1) の KKT は、x_i>0 なら ∇f_i = μ、
        //    x_i=0 なら ∇f_i ≥ μ。μ は「x_i>0 部分の勾配平均」を簡易近似として計算。
        vector<int> active;
        active.reserve(n);
        for(int i = 0; i < n; ++i) {
            if(x[i] > 1e-12) active.push_back(i);
        }
        if(active.empty()) {
            // 全て x_i ≈ 0 ならスキップして μ=0 とみなす
            active.push_back(0);
        }
        double mu = 0.0;
        for(int idx : active) {
            mu += g[idx];
        }
        mu /= active.size();

        double kkt_res = 0.0;
        for(int i = 0; i < n; ++i) {
            if(x[i] > 1e-12) {
                // x_i>0: ∇f_i - μ の絶対値をチェック
                kkt_res = max(kkt_res, fabs(g[i] - mu));
            } else {
                // x_i=0: ∇f_i ≥ μ を確認
                double tmp = max(0.0, mu - g[i]);
                kkt_res = max(kkt_res, tmp);
            }
        }
        if(kkt_res < tol) {
            // 収束
            // cout << "[PGD] iter=" << iter << "  KKT_res=" << kkt_res << "\n";
            return true;
        }

        // (f) BBステップ幅更新 (s = x-x_prev, y = g-g_prev)
        s = x - x_prev;
        y_vec = g - g_prev;
        double sty = s.dot(y_vec);
        double sts = s.squaredNorm();
        double alpha_bb;
        if(sty > 1e-16) {
            alpha_bb = sts / sty;
        } else {
            alpha_bb = alpha_min;
        }
        alpha = min(max(alpha_bb, alpha_min), alpha_max);
    }

    // max_iter に到達しても収束せず
    return false;
}

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

vector<vector<int>> construct_subsets(int size, int k) {
    vector<vector<int>> subsets;
    vector<int> comb(size);
    function<void(int, int)> dfs = [&](int start, int depth) {
        if(depth == size) {
            subsets.emplace_back(comb.begin(), comb.end());
            return;
        }
        for(int x = start; x < k; x++) {
            comb[depth] = x;
            dfs(x + 1, depth + 1);
        }
    };
    dfs(0, 0);

    return subsets;
}

// =======================
// 疑似逆行列を求めて射影
// =======================

// int main() {
//     Input input = parse_input();

//     const int COMB_K = 4;
//     auto subsets = construct_subsets(COMB_K, input.K);

//     TimeKeeper timer(100.0);

//     vector<double> best_errs(input.H);
//     for(int h = 0; h < input.H; ++h) {
//         vector<double> true_errors;
//         for(auto &subset : subsets) {
//             Eigen::MatrixXd A;
//             A.resize(3, COMB_K);
//             for(int k = 0; k < COMB_K; ++k) {
//                 auto col = input.own[subset[k]];
//                 Eigen::Vector3d c(col[0], col[1], col[2]);
//                 A.col(k) = c;
//             }

//             Eigen::Vector3d t;
//             t(0) = input.target[h][0];
//             t(1) = input.target[h][1];
//             t(2) = input.target[h][2];

//             // 制約なし最小二乗（Asubが正則なら Asub.inverse()*t, そうでなければsolve）
//             Eigen::VectorXd w_sub = A.colPivHouseholderQr().solve(t);
//             w_sub = ProjectOntoSimplex(w_sub);

//             Eigen::Vector3d c_hat = A * w_sub;
//             double sq_error_ext = (t - c_hat).squaredNorm();
//             double sq_error_rgb =
//                 (Eigen::Vector3d(c_hat(0), c_hat(1), c_hat(2)) - Eigen::Vector3d(input.target[h][0], input.target[h][1], input.target[h][2])).squaredNorm();

//             double sum_w = w_sub.sum();
//             // assert(abs(sum_w - 1.0) < 1e-6);

//             vector<double> weights(COMB_K);
//             int nonw_zero_count = 0;
//             for(int i = 0; i < COMB_K; ++i) {
//                 weights[i] = w_sub(i) / sum_w;
//                 if(weights[i] > 1e-6) {
//                     nonw_zero_count++;
//                 }
//             }

//             double sum_new_w = accumulate(weights.begin(), weights.end(), 0.0);
//             double true_err = true_error(input, weights, subset, input.target[h]);

//             if(!isfinite(true_err)) {
//                 // nan, inf
//                 continue;
//             }

//             true_errors.push_back(true_err);
//         }
//         double min_true_err = *min_element(true_errors.begin(), true_errors.end());
//         cpp_dump(h, min_true_err * 1e4);
//         best_errs[h] = min_true_err;
//     }
// }

// =======================
// アクティブセットの全サブセット + 射影（誤差が残る）
// =======================

int main() {
    Input input = parse_input();

    const int COMB_K = 4;
    auto subsets = construct_subsets(COMB_K, input.K);

    TimeKeeper timer(100.0);

    vector<double> best_errs(input.H);
    for(int h = 0; h < input.H; ++h) {
        vector<double> true_errors;
        for(auto &subset : subsets) {
            Eigen::MatrixXd A_ext;
            A_ext.resize(3, COMB_K);
            for(int k = 0; k < COMB_K; ++k) {
                auto col = input.own[subset[k]];
                Eigen::Vector3d c(col[0], col[1], col[2]);
                A_ext.block<3, 1>(0, k) = c; // 上3行
            }

            Eigen::NNLS<Eigen::MatrixXd> nnls_solver;
            nnls_solver.compute(A_ext);
            nnls_solver.setTolerance(1e-7);
            nnls_solver.setMaxIterations(50);

            Eigen::Vector3d t_ext;
            t_ext(0) = input.target[h][0];
            t_ext(1) = input.target[h][1];
            t_ext(2) = input.target[h][2];

            Eigen::VectorXd x = nnls_solver.solve(t_ext);

            Eigen::Vector3d c_hat_ext = A_ext * x;
            double sq_error_ext = (t_ext - c_hat_ext).squaredNorm();
            // double sq_error_rgb =
            //     (Eigen::Vector3d(c_hat_ext(0), c_hat_ext(1), c_hat_ext(2)) - Eigen::Vector3d(input.target[h][0], input.target[h][1], input.target[h][2]))
            //         .squaredNorm();

            x = ProjectOntoSimplex(x); // 重みを単体に投影して正規化する
            double sum_w = x.sum();
            assert(abs(sum_w - 1.0) < 1e-6);

            vector<double> weights(COMB_K);
            int nonw_zero_count = 0;
            for(int i = 0; i < COMB_K; ++i) {
                weights[i] = x(i) / sum_w;
                if(weights[i] > 1e-6) {
                    nonw_zero_count++;
                }
            }

            double sum_new_w = accumulate(weights.begin(), weights.end(), 0.0);
            double true_err = true_error(input, weights, subset, input.target[h]);
            // assert(abs(true_err - sq_error_ext) < 1e-6); // true_err と sq_error_ext が一致することを確認

            // cpp_dump(subset, true_err);

            true_errors.push_back(true_err);
        }
        double min_true_err = *min_element(true_errors.begin(), true_errors.end());
        cpp_dump(h, min_true_err * 1e4);
        best_errs[h] = min_true_err;
    }

    // for(auto &te : best_errs) {
    //     cerr << boost::format("%.4f ") % (te * 1e4);
    // }

    // cerr << endl;

    return 0;
}

// =======================
// PGDの全サブセット.PDG内で射影
// =======================
// bool frank_wolfe_simplex_ls(const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
//                             Eigen::VectorXd &x, // 初期値と解 (size n)
//                             double tol = 1e-10, int max_iter = 10000) {
//     const int n = x.size();
//     const Eigen::MatrixXd ATA = A.transpose() * A;
//     const Eigen::VectorXd ATb = A.transpose() * b;

//     for(int iter = 0; iter < max_iter; ++iter) {
//         // 勾配
//         Eigen::VectorXd g = ATA * x - ATb;

//         // 最も小さい勾配成分のインデックス
//         int s_idx;
//         g.minCoeff(&s_idx);

//         // s: 標準基底ベクトル（その成分だけ1）
//         Eigen::VectorXd s = Eigen::VectorXd::Zero(n);
//         s[s_idx] = 1.0;

//         // ステップ方向
//         Eigen::VectorXd d = s - x;

//         // 最適ステップサイズγを解析的に計算
//         Eigen::VectorXd Ad = A * d;
//         Eigen::VectorXd Ax = A * x;
//         double num = (Ax - b).dot(Ad);
//         double denom = Ad.squaredNorm();
//         double gamma = (denom == 0.0) ? 0.0 : std::max(0.0, std::min(1.0, -num / denom));

//         // x を更新
//         x += gamma * d;

//         // 収束判定（KKT: d^T grad = g[s_idx] - g^T x = 最小勾配と現在xの勾配差）
//         double gap = g.dot(x) - g[s_idx];
//         // if(fabs(gap) < tol) {
//         //     return true;
//         // }
//     }
//     return false;
// }

// int main() {
//     Input input = parse_input();

//     const int COMB_K = 4;
//     auto subsets = construct_subsets(COMB_K, input.K);

//     TimeKeeper timer(100.0);

//     vector<double> best_errs(input.H);
//     for(int h = 0; h < input.H; ++h) {
//         vector<double> true_errors;
//         for(auto &subset : subsets) {
//             Eigen::MatrixXd A;
//             A.resize(3, COMB_K);
//             for(int k = 0; k < COMB_K; ++k) {
//                 auto col = input.own[subset[k]];
//                 Eigen::Vector3d c(col[0], col[1], col[2]);
//             }

//             Eigen::Vector3d t;
//             t(0) = input.target[h][0];
//             t(1) = input.target[h][1];
//             t(2) = input.target[h][2];

//             // 一様分布に初期化しておく
//             Eigen::VectorXd x;
//             x.resize(COMB_K);
//             x.setOnes();
//             x /= COMB_K;

//             bool ok = frank_wolfe_simplex_ls(A, t, x, 1e-12, 1000000000);

//             Eigen::Vector3d c_hat = A * x;
//             double sq_error_ext = (t - c_hat).squaredNorm();
//             double sq_error_rgb =
//                 (Eigen::Vector3d(c_hat(0), c_hat(1), c_hat(2)) - Eigen::Vector3d(input.target[h][0], input.target[h][1], input.target[h][2])).squaredNorm();

//             double sum_w = x.sum();
//             assert(abs(sum_w - 1.0) < 1e-6);

//             vector<double> weights(COMB_K);
//             int nonw_zero_count = 0;
//             for(int i = 0; i < COMB_K; ++i) {
//                 weights[i] = x(i) / sum_w;
//                 if(weights[i] > 1e-6) {
//                     nonw_zero_count++;
//                 }
//             }

//             double sum_new_w = accumulate(weights.begin(), weights.end(), 0.0);
//             assert(abs(sum_new_w - 1.0) < 1e-6); // 重みの合計が 1 に近いことを確認
//             double true_err = true_error(input, weights, subset, input.target[h]);

//             true_errors.push_back(true_err);
//         }
//         double min_true_err = *min_element(true_errors.begin(), true_errors.end());
//         cpp_dump(h, min_true_err * 1e4);
//         best_errs[h] = min_true_err;
//     }

//     cpp_dump(input.K, subsets.size());

//     return 0;
// }

// =======================
// アクティブセットの全サブセット + 合計1に正規化する4次元（なぜか誤差がでる。行列の条件が違うとだめらしい。）
// =======================
// int main() {
//     Input input = parse_input();

//     const int COMB_K = 4;
//     auto subsets = construct_subsets(COMB_K, input.K);

//     TimeKeeper timer(100.0);

//     vector<double> best_errs(input.H);
//     for(int h = 0; h < input.H; ++h) {
//         vector<double> true_errors;
//         vector<int> best_subset;
//         double best_error = 1e9;
//         for(auto &subset : subsets) {
//             Eigen::MatrixXd A_ext;
//             A_ext.resize(4, COMB_K);
//             for(int k = 0; k < COMB_K; ++k) {
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
//             Eigen::Vector4d c_hat_ext = A_ext * x;
//             double sq_error_ext = (t_ext - c_hat_ext).squaredNorm();
//             double sq_error_rgb =
//                 (Eigen::Vector3d(c_hat_ext(0), c_hat_ext(1), c_hat_ext(2)) - Eigen::Vector3d(input.target[h][0], input.target[h][1], input.target[h][2]))
//                     .squaredNorm();

//             double sum_w = 0.0;
//             for(int i = 0; i < COMB_K; ++i) {
//                 sum_w += x(i);
//             }
//             // if(abs(sum_w - 1.0) > 1e-6) {
//             //     x = ProjectOntoSimplex(x); // 重みを単体に投影して正規化する
//             // }
//             // assert(abs(sum_w - 1.0) < 1e-4);

//             vector<double> weights(COMB_K);
//             int nonw_zero_count = 0;
//             for(int i = 0; i < COMB_K; ++i) {
//                 weights[i] = x(i) / sum_w;
//                 if(weights[i] > 1e-6) {
//                     nonw_zero_count++;
//                 }
//             }

//             double sum_new_w = accumulate(weights.begin(), weights.end(), 0.0);
//             if(abs(sum_new_w - 1.0) > 1e-6) {
//                 cerr << "Error: 重みの合計が 1 に近くありません。sum_new_w = " << sum_new_w << endl;
//                 assert(false);
//             }
//             double true_err = true_error(input, weights, subset, input.target[h]);

//             true_errors.push_back(true_err);
//             if(true_err < best_error) {
//                 best_error = true_err;
//                 best_subset = subset;
//             }
//         }
//         double min_true_err = *min_element(true_errors.begin(), true_errors.end());
//         best_errs[h] = min_true_err;

//         Eigen::MatrixXd A_ext;
//         A_ext.resize(4, COMB_K);
//         for(int k = 0; k < COMB_K; ++k) {
//             auto col = input.own[best_subset[k]];
//             Eigen::Vector3d c(col[0], col[1], col[2]);
//             A_ext.block<3, 1>(0, k) = c; // 上3行
//         }
//         A_ext.row(3).setOnes();

//         Eigen::JacobiSVD<Eigen::MatrixXd> svd(A_ext);
//         double cond = svd.singularValues()(0) / svd.singularValues()(3);

//         cpp_dump(h, min_true_err * 1e4, cond);
//     }

//     cpp_dump(input.K, subsets.size());

//     // for(auto &te : best_errs) {
//     //     cout << boost::format("%.4f ") % (te * 1e4);
//     // }
//     // cout << endl;

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

// =======================
// アクティブセット + 合計1に射影（射影関数を通しても、単純な割り算でも誤差がでる）
// =======================
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

//     double sum_true_err = 0.0;
//     for(int h : range(input.H)) {
//         Eigen::NNLS<Eigen::MatrixXd> nnls_solver;
//         nnls_solver.compute(A);

//         nnls_solver.setTolerance(1e-7);
//         nnls_solver.setMaxIterations(50);
//         Eigen::Vector3d t(input.target[h][0], input.target[h][1], input.target[h][2]);
//         Eigen::VectorXd x = nnls_solver.solve(t);
//         Eigen::Vector3d c_hat = A * x;
//         double sq_error = (t - c_hat).squaredNorm();
//         // x = ProjectOntoSimplex(x); // 重みを単体に投影して正規化する
//         double sum_w = x.sum();
//         vector<double> x_vec(x.data(), x.data() + x.size());
//         // cpp_dump(x_vec, sum_w);
//         // assert(abs(sum_w - 1.0) < 1e-6); // 重みの合計が 1 に近いことを確認
//         vector<double> weights(input.K);
//         for(int i = 0; i < input.K; ++i) {
//             weights[i] = x(i) / sum_w; // 重みを正規化
//         }
//         double sum_new_w = accumulate(weights.begin(), weights.end(), 0.0);
//         assert(abs(sum_new_w - 1.0) < 1e-6); // 重みの合計が 1 に近いことを確認
//         double true_err = true_error(input, weights, indices, input.target[h]);
//         double diff = abs(sq_error - true_err);
//         cerr << boost::format(" %d: sq_error = %.4f, true_err = %.4f, diff = %.4f\n") % h % (sq_error * 1e4) % (true_err * 1e4) % (diff * 1e4);

//         sum_true_err += true_err;
//         // cerr << " " << h << ": " << sq_error * 1e4 << "\n";
//     }
//     cpp_dump(sum_true_err);
//     cerr << "Total time: " << timer.getElapsedTime() << " seconds.\n";

//     return 0;
// }

// =======================
// アクティブセット
// =======================
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
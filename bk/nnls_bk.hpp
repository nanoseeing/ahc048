
#pragma once

#include "common.hpp"
#include "utils.hpp"

// ====================================
// NNLSを解くためのクラス
// ====================================

#include <Eigen/Core>
#include <Eigen/Dense>

// 単純体への射影関数
Eigen::VectorXd ProjectOntoSimplex(const Eigen::VectorXd& v) {
    const int n = v.size();
    std::vector<double> u(n);
    for(int i = 0; i < n; ++i)
        u[i] = v[i];
    std::sort(u.begin(), u.end(), std::greater<double>());

    std::vector<double> cumsum(n);
    cumsum[0] = u[0];
    for(int i = 1; i < n; ++i)
        cumsum[i] = cumsum[i - 1] + u[i];

    int rho = -1;
    double theta = 0;
    for(int j = 0; j < n; ++j) {
        double t = (cumsum[j] - 1.0) / (j + 1);
        if(u[j] - t > 0) {
            rho = j;
            theta = t;
        }
    }
    if(rho < 0) {
        return Eigen::VectorXd::Constant(n, 1.0 / n);
    }
    Eigen::VectorXd w(n);
    for(int i = 0; i < n; ++i) {
        w[i] = std::max(v[i] - theta, 0.0);
    }
    return w;
}

// -----------------------------------------------------------------------------
// estimateMaxEigenvalue()
//   パワーイテレーションにより、ATA = Aᵀ A の最大固有値を推定する。
//   A: (m×n) 行列、powerIter: イテレーション回数（10～20程度で十分）
// -----------------------------------------------------------------------------
double estimateMaxEigenvalue(const Eigen::MatrixXd& A, int powerIter = 20) {
    const int n = A.cols();
    // Aᵀ A に対するパワー法
    Eigen::VectorXd v = Eigen::VectorXd::Random(n);
    v.normalize();
    for(int it = 0; it < powerIter; ++it) {
        // w ← (Aᵀ A) v
        Eigen::VectorXd w = A.transpose() * (A * v);
        double wnorm = w.norm();
        if(wnorm <= 0) break;
        v = w / wnorm;
    }
    // λ ≈ vᵀ (Aᵀ A) v
    Eigen::VectorXd Av = A * v;
    Eigen::VectorXd ATAv = A.transpose() * Av;
    double lambda = v.dot(ATAv);
    return lambda;
}

// -----------------------------------------------------------------------------
// nnls_projected_gradient_bb_clipped()
//   A ∈ R^{m×n}, b ∈ R^m を与えて、
//   min_{x ∈ simplex} ½ ||A x − b||^2 をクリップ付き BB ステップ幅で解く。
//   ・x は「x_i >=0, sum_i x_i = 1」を常に満たす（ProjectOntoSimplexを挟む）。
//   tol: KKT 条件残差許容値
//   max_iter: イテレーション上限
// -----------------------------------------------------------------------------
bool nnls_projected_gradient_bb_clipped(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                        Eigen::VectorXd& x, // 初期 guess を与え、解がここに返る (size n)
                                        double tol = 1e-7,  // KKT 条件の残差閾値
                                        int max_iter = 1e3, // 最大イテレーション数
                                        bool is_alpha_max_fixed = true) {
    const int m = A.rows();
    const int n = A.cols();
    if(b.size() != m || x.size() != n) {
        cerr << "[nnls_pg_bb] サイズ不一致: A(" << m << "×" << n << "), b(" << b.size() << "), x(" << x.size() << ")\n";
        return false;
    }

    // 1) 事前計算: ATA, ATb
    Eigen::MatrixXd ATA = A.transpose() * A; // (n×n)
    Eigen::VectorXd ATb = A.transpose() * b; // (n)

    // 2) λ_max = 最大固有値(AᵀA) を推定（パワー法）
    double lambda_max = estimateMaxEigenvalue(A, 20);
    if(lambda_max <= 0) lambda_max = 1e-3; // 念のためゼロ割回避

    // 3) α_max, α_min の設定（上限・下限をクリップする）
    //    上限: 0.8 / λ_max  (「1/λ_max の約80%」)
    //    下限: 1e-6 / λ_max
    double alpha_max;
    if(is_alpha_max_fixed) {
        alpha_max = 1e8; // alphaがでかいと発散することがあるので注意
    } else {
        alpha_max = 0.99 / lambda_max;
    }
    const double alpha_min = 1e-12 / lambda_max;

    // 4) 初期化: x >= 0 且つ sum(x)=1 にする
    //    └  もし呼び出し側が x ≥ 0, sum=1 を用意していなければ、
    //        ここで一様分布に初期化してもよい。
    x = ProjectOntoSimplex(x);

    // 5) 初期勾配 g = ATA*x - ATb
    Eigen::VectorXd g = ATA * x - ATb;

    // 6) 初期 step size: 1 / λ_max
    double alpha = 1.0 / lambda_max;

    // (反復用テンポラリ)
    Eigen::VectorXd x_prev(n), g_prev(n), s(n), y_vec(n);

    for(int iter = 0; iter < max_iter; ++iter) {
        // (a) 前回の保存
        x_prev = x;
        g_prev = g;

        // (b) 勾配ステップ
        Eigen::VectorXd x_tent = x - alpha * g;

        // (c) 単純体への射影 (sum=1, x_i>=0 を維持)
        x = ProjectOntoSimplex(x_tent);

        // (d) 勾配の再計算
        g = ATA * x - ATb;

        // (e) KKT 条件による収束判定
        //       「最適性残差 = max_i |min(x_i, g_i)|」が tol 未満なら終了
        double kkt_res = 0.0;
        for(int i = 0; i < n; ++i) {
            double xi = x[i], gi = g[i];
            // x_i > 0 なら g_i ≈ 0、x_i = 0 なら g_i ≥ 0
            double tmp = std::min(xi, gi);
            kkt_res = std::max(kkt_res, std::abs(tmp));
        }
        if(kkt_res < tol) {
            // 収束した
            // cout << "[BB-PGD] iter=" << iter << "  KKT_res=" << kkt_res << "\n";
            return true;
        }

        // (f) BBステップ幅更新
        s = x - x_prev;     // Δx
        y_vec = g - g_prev; // Δg
        double sty = s.dot(y_vec);
        double sts = s.squaredNorm();
        double alpha_bb;
        if(sty > 1e-16) {
            alpha_bb = sts / sty;
        } else {
            // 分母が非常に小さい・負になるときは小さな α_min を使っておく
            alpha_bb = alpha_min;
        }
        // (g) α をクリップ
        alpha = std::min(std::max(alpha_bb, alpha_min), alpha_max);
    }

    // max_iter に到達しても収束せず
    return false;
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

vector<vector<double>> Gram;
vector<vector<double>> pseudo;
vector<vector<double>> invG;

class ColorMixer {
  public:
    struct Result {
        double cost;
        vector<int> indices;
        vector<double> weights;

        bool operator<(Result const& o) const {
            return cost < o.cost;
        }
    };

    struct SubsetInfo {
        int size;
        vector<int> indices;
        vector<vector<double>> Gram;   // Gram 行列: size×size
        vector<vector<double>> pseudo; // 擬似逆行列: size×3
    };

    static constexpr double EPS = 1e-7;         // 許容誤差 (sum_w ≈ 1.0 ± epsに収束)
    static constexpr int MAX_ITER = 50;         // 簡易評価
    static constexpr int MAX_ITER_HEAVY = 1000; // 最大反復回数

    vector<Color> paints;
    int K;

    ColorMixer(const vector<Color>& paints_input) : paints(paints_input) {
        K = paints.size();
    }

    vector<Result> solve_nnls(const Color& t, int comb_size, int find_top_n) {
        auto subsets = construct_subsets(comb_size, this->K);
        sort(ALL(subsets), [&](auto& a, auto& b) { return xor_rng.next() < 0.5; });

        const int MAX_SUBSETS = 500;
        if(subsets.size() > MAX_SUBSETS) {
            subsets.resize(MAX_SUBSETS);
        }

        // const int TEMP_HEAP_SIZE = 30;

        // priority_queue<Result> heap;
        // for(const auto& indices : subsets) {
        //     Result r = solve_nnls_inv(t, indices);
        //     if((int)heap.size() < find_top_n) {
        //         heap.push(r);
        //     } else if(r.err < heap.top().err) {
        //         heap.pop();
        //         heap.push(r);
        //     }
        // }
        // vector<Result> results;
        // while(!heap.empty()) {
        //     auto r = heap.top();
        //     heap.pop();
        //     results.push_back(r);
        //     Result r2 = solve_nnls_pdm(r.indices, t, true, EPS, MAX_ITER);
        //     if(r2.err < r.err) {
        //         results.push_back(r2);
        //     } else {
        //         results.push_back(r);
        //     }
        // }
        // sort(results.begin(), results.end(), [](const Result& a, const Result& b) { return a.err < b.err; });

        vector<Result> results;
        for(const auto& indices : subsets) {
            Result r = solve_nnls_pdm(indices, t, true, EPS, MAX_ITER);
            results.emplace_back(move(r));
        }
        sort(results.begin(), results.end(), [](const Result& a, const Result& b) { return a.cost < b.cost; });
        results.resize(min(find_top_n, (int)results.size()));

        const int MAX_HEAVY_NNLS = 10;
        for(int i : range(min((int)results.size(), MAX_HEAVY_NNLS))) {
            auto& r = results[i];
            if(r.cost < 1e-4) continue;
            Result r3 = solve_nnls_pdm(r.indices, t, false, EPS, MAX_ITER_HEAVY);
            if(r3.cost < r.cost) {
                results[i] = r3;
            }
        }
        return results;
    }

    Result solve_nnls_pdm(const vector<int>& indices, const Color& t_color, double is_alpha_max_fixed = true, double eps = EPS, int max_iter = MAX_ITER) {
        int n = static_cast<int>(indices.size());
        Eigen::Vector3d t(t_color[0], t_color[1], t_color[2]);
        Eigen::VectorXd x0 = Eigen::VectorXd::Constant(n, 1.0 / double(n));
        Eigen::MatrixXd A;

        A.resize(3, n);
        for(int j = 0; j < n; ++j) {
            int paint_idx = indices[j];
            const auto& col = paints[paint_idx];
            Eigen::Vector3d c(col[0], col[1], col[2]);
            A.col(j) = c;
        }

        Eigen::VectorXd w;

        bool ok = nnls_projected_gradient_bb_clipped(A, t, x0, eps, max_iter, is_alpha_max_fixed);
        auto err = (A * x0 - t).norm();
        w = x0.transpose();

        double sum_w = w.sum();
        assert(abs(sum_w - 1.0) < 1e-6); // 合計が 1 に正規化されていることを確認

        vector<double> weights;
        for(int i = 0; i < n; ++i) {
            weights.push_back(w(i));
        };

        return Result{err, indices, weights};
    }

    Result solve_nnls_inv(const Color& t, vector<int> indices) {
        int n = static_cast<int>(indices.size());

        calc_gram_inv(indices);

        double t_norm2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];

        // 1) 擬似逆行列 × t で制約なし最小二乗解を得る
        vector<double> w_ls(n, 0.0);
        for(int i = 0; i < n; i++) {
            // pseudo はサイズ n×3 の行列
            w_ls[i] = pseudo[i][0] * t[0] + pseudo[i][1] * t[1] + pseudo[i][2] * t[2];
        }

        // 2) クリッピング＆正規化 (w_ls を非負化し、合計 = 1 にする)
        double sum = 0.0;
        for(int i = 0; i < n; i++) {
            if(w_ls[i] < 0.0) w_ls[i] = 0.0;
            sum += w_ls[i];
        }
        if(sum <= 0.0) {
            // 全部 0 になったら一様分配
            double uni = 1.0 / n;
            for(int i = 0; i < n; i++) {
                w_ls[i] = uni;
            }
        } else {
            for(int i = 0; i < n; i++) {
                w_ls[i] /= sum;
            }
        }

        // 3) b = A_S^T * t を計算
        vector<double> b(n, 0.0);
        for(int i = 0; i < n; i++) {
            int pk = indices[i];
            b[i] = paints[pk][0] * t[0] + paints[pk][1] * t[1] + paints[pk][2] * t[2];
        }

        // 4) w^T G w と -2 b^T w を計算
        double wGw = 0.0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                wGw += w_ls[i] * Gram[i][j] * w_ls[j];
            }
        }
        double bTw = 0.0;
        for(int i = 0; i < n; i++) {
            bTw += b[i] * w_ls[i];
        }
        double eprime = wGw - 2.0 * bTw;

        // 5) 二乗誤差 = eprime + t_norm2
        double true_err = eprime + t_norm2;

        return Result{true_err, indices, w_ls};
    }

    void invertMatrix(int size) const {
        // tmp は size × (2*size) の拡大行列 [G | I]
        vector<vector<double>> tmp(size, vector<double>(2 * size, 0.0));
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                tmp[i][j] = Gram[i][j];
            }
            for(int j = 0; j < size; j++) {
                tmp[i][size + j] = (i == j ? 1.0 : 0.0);
            }
        }
        // Gauss-Jordan
        for(int i = 0; i < size; i++) {
            // ピボット選択
            int pivot = i;
            for(int r = i + 1; r < size; r++) {
                if(fabs(tmp[r][i]) > fabs(tmp[pivot][i])) {
                    pivot = r;
                }
            }
            if(pivot != i) {
                swap(tmp[i], tmp[pivot]);
            }
            double diag = tmp[i][i];
            if(fabs(diag) < 1e-12) {
                diag = (diag >= 0 ? 1e-12 : -1e-12);
                tmp[i][i] = diag;
            }
            double invDiag = 1.0 / tmp[i][i];
            for(int c = 0; c < 2 * size; c++) {
                tmp[i][c] *= invDiag;
            }
            for(int r = 0; r < size; r++) {
                if(r == i) continue;
                double factor = tmp[r][i];
                if(fabs(factor) < 1e-16) continue;
                for(int c = 0; c < 2 * size; c++) {
                    tmp[r][c] -= factor * tmp[i][c];
                }
            }
        }
        // 右半分が逆行列
        invG.assign(size, vector<double>(size, 0.0));
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                invG[i][j] = tmp[i][size + j];
            }
        }
    }

    void calc_gram_inv(vector<int> const& comb) {
        int size = static_cast<int>(comb.size());

        Gram.assign(size, vector<double>(size, 0.0));
        for(int i = 0; i < size; i++) {
            for(int j = i; j < size; j++) {
                double dot = paints[comb[i]][0] * paints[comb[j]][0] + paints[comb[i]][1] * paints[comb[j]][1] + paints[comb[i]][2] * paints[comb[j]][2];
                Gram[i][j] = dot;
                if(i != j) Gram[j][i] = dot;
            }
        }

        // Gram の逆行列 invG を計算
        invertMatrix(size);

        // 擬似逆行列 = invG × A_S^T (sz×3)
        pseudo.assign(size, vector<double>(3, 0.0));
        for(int i = 0; i < size; i++) {
            for(int d = 0; d < 3; d++) {
                double sum = 0.0;
                for(int j = 0; j < size; j++) {
                    sum += invG[i][j] * paints[comb[j]][d];
                }
                pseudo[i][d] = sum;
            }
        }
    }
};

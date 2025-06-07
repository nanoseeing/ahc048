
#pragma once

#include "common.hpp"
#include "ex/nnls.hpp"
#include "game.hpp"
#include "utils.hpp"

// ====================================
// NNLSを解くためのクラス
// ====================================

std::mt19937 engine(42);

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

class ColorMixer {
  public:
    struct Result {
        double err;
        vector<int> indices;
        vector<double> weights;

        bool operator<(Result const& o) const {
            return err < o.err;
        }
    };
    struct SubsetInfo {
        int size;
        vector<int> indices;
    };

    Input& input;
    unordered_map<pair<int, int>, vector<Result>> results_cache; // key:(h, comb_size), value: Result

    static constexpr double EPS = 1e-7;
    static constexpr int MAX_ITER = 30;
    const int SUBSET_NUM_THRESHOLD = 500; // 20C2 = 190, 20C3 = 1140, 20C4 = 4845

    ColorMixer(Input& input_) : input(input_) {
        construct();
    }

    vector<Result> get_results(int h, int comb_size) {
        assert(0 <= h && h < input.H);
        assert(2 <= comb_size && comb_size <= 4);
        pair<int, int> key = {h, comb_size};
        return results_cache[key];
    }

    void construct() {
        const int FIND_TOP_N = 100;

        for(int comb_size = 2; comb_size <= 4; ++comb_size) {
            auto subsets = construct_subsets(comb_size, input.K);
            if((int)subsets.size() > SUBSET_NUM_THRESHOLD) {
                shuffle(subsets.begin(), subsets.end(), engine);
                subsets.resize(min((int)SUBSET_NUM_THRESHOLD, (int)subsets.size()));
            }
            for(int h = 0; h < input.H; ++h) {
                Color& t = input.target[h];
                vector<Result> results;

                if(comb_size == 4) {
                    // NNLSを解けば基本的に4色だけ残るはず。
                    vector<int> indices;
                    for(int i = 0; i < this->input.K; ++i) {
                        indices.push_back(i);
                    }
                    Result r = nnls(t, indices, EPS, MAX_ITER);

                    vector<int> inds4;
                    vector<double> weights4;
                    for(int i = 0; i < this->input.K; ++i) {
                        if(r.weights[i] > EPS) {
                            inds4.push_back(i);
                            weights4.push_back(r.weights[i]);
                        }
                    }
                    assert((int)inds4.size() <= 4);

                    // 一応計算Errorは計算しなおさないといけないが、ほぼ誤差の範囲のはず
                    Result new_r = Result{r.err, move(inds4), move(weights4)};
                    results.emplace_back(move(new_r));
                }

                // 2, 3色のNNLSを解く
                for(auto& indices : subsets) {
                    Result r = nnls(t, indices, EPS, MAX_ITER);
                    results.emplace_back(move(r));
                }
                sort(ALL(results), [&](auto& a, auto& b) { return a.err < b.err; });
                results.resize(min(FIND_TOP_N, (int)results.size()));
                results_cache[{h, comb_size}] = move(results);
            }
        }
    }

    Result nnls(Color& target, vector<int>& indices, double tol, double iter) {
        const int N = indices.size();

        Eigen::MatrixXd A_ext;
        A_ext.resize(4, N);
        for(int k = 0; k < N; ++k) {
            auto col = this->input.own[indices[k]];
            Eigen::Vector3d c(col[0], col[1], col[2]);
            A_ext.block<3, 1>(0, k) = c;
        }
        A_ext.row(3).setOnes();

        Eigen::NNLS<Eigen::MatrixXd> nnls_solver;
        nnls_solver.compute(A_ext);
        nnls_solver.setTolerance(tol);
        nnls_solver.setMaxIterations(iter);

        Eigen::Vector4d t_ext;
        t_ext(0) = target[0];
        t_ext(1) = target[1];
        t_ext(2) = target[2];
        t_ext(3) = 1.0; // 「和が１になる」項を擬似的に加える

        Eigen::VectorXd x = nnls_solver.solve(t_ext);
        x = ProjectOntoSimplex(x); // 射影して非負かつ合計が1にする

        double sum_w = x.sum();
        assert(abs(sum_w - 1.0) < 1e-6);

        vector<double> weights;
        for(int i = 0; i < N; ++i) {
            weights.push_back(x(i) / sum_w); // 射影すれば1になるはずだが念のため正規化しておく
        }

        double true_err = calc_true_error(weights, indices, target);
        return Result{true_err, indices, weights};
    }

    double calc_true_error(vector<double>& weights, vector<int>& indices, Color& target) {
        double true_err = 0.0;
        for(int j = 0; j < 3; ++j) {
            double now_c = 0.0;
            for(int i = 0; i < (int)indices.size(); ++i) {
                int idx = indices[i];
                now_c += input.own[idx][j] * weights[i];
            }
            double diff = now_c - target[j];
            true_err += diff * diff;
        }
        return sqrt(true_err);
    }
};

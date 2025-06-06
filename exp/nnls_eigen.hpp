#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>

#include "common.hpp"
#include "ex/nnls.hpp"
#include "utils.hpp"

// =========================================================
// EigenComb
// =========================================================

using EigenColor = Eigen::Vector3d;

tuple<Eigen::VectorXd, double, EigenColor> solve_nnls(const Eigen::MatrixXd& A, const EigenColor& t, double eps = 1e-15) {
    int n = A.cols();

    // NNLSを構築して解く
    Eigen::NNLS<Eigen::MatrixXd> nnls_solver;
    nnls_solver.compute(A);
    nnls_solver.setTolerance(eps);
    nnls_solver.setMaxIterations(100 * n);
    Eigen::VectorXd w = nnls_solver.solve(t);

    // 非負制約解が収束しなかった場合は一様分配しておく
    if(nnls_solver.info() != Eigen::ComputationInfo::Success) {
        double uni = 1.0 / n;
        w = Eigen::VectorXd::Constant(n, uni);
    }

    // 合計1に正規化
    double sumw = w.sum();
    if(sumw <= 1e-12) {
        double uni = 1.0 / n;
        for(int i = 0; i < n; i++)
            w(i) = uni;
        sumw = 1.0;
    } else {
        w /= sumw;
    }

    EigenColor c_hat = A * w;
    double true_err = (t - c_hat).squaredNorm();

    return {w, true_err, c_hat};
}

class ColorMixer {
  public:
    struct Result {
        double squared_error;
        vector<int> indices;
        vector<double> weights;
    };

    struct HeapItem {
        double err;
        int subset_idx;
        vector<double> weights;

        bool operator<(HeapItem const& o) const {
            return err < o.err;
        }
    };

    ColorMixer(const vector<EigenColor>& paints_input) : paints(paints_input) {
        K = static_cast<int>(paints.size());
        prepare_subsets();
    }

    ColorMixer(const vector<Color>& paints_input, const vector<int>& _comb_size) : comb_size(_comb_size) {
        for(const auto& col : paints_input) {
            EigenColor eigin_col = EigenColor(col[0], col[1], col[2]);
            paints.emplace_back(eigin_col);
        }

        K = static_cast<int>(paints.size());
        prepare_subsets();
    }

    tuple<Eigen::VectorXd, double, EigenColor> solve_nnls_for_indices(const vector<int>& indices, const EigenColor& t) const {
        int n = static_cast<int>(indices.size());
        Eigen::Matrix<double, 3, Eigen::Dynamic> A(3, n);
        for(int j = 0; j < n; j++) {
            A.col(j) = paints[indices[j]];
        }
        return solve_nnls(A, t);
    }

    vector<Result> find_topN(const Color& t, int topN) const {
        EigenColor t_eigen(t[0], t[1], t[2]);
        return find_topN(t_eigen, topN);
    }

    vector<Result> find_topN(const EigenColor& t, int topN) const {
        priority_queue<HeapItem> heap;
        int total = static_cast<int>(subsets.size());

        for(int si = 0; si < total; si++) {
            const SubsetInfo& info = subsets[si];
            int n = static_cast<int>(info.indices.size()); // 2,3,4 のいずれか
            auto [w, true_err, c_hat] = solve_nnls_for_indices(info.indices, t);

            vector<double> retw;
            for(int i = 0; i < n; i++) {
                retw.push_back(w(i));
            }

            // ヒープに突っ込む (topN を維持)
            if(static_cast<int>(heap.size()) < topN) {
                heap.push({true_err, si, retw});
            } else if(true_err < heap.top().err) {
                heap.pop();
                heap.push({true_err, si, retw});
            }
        }

        // ヒープに残った topN 件を vector<Result> にまとめて返す
        int M = static_cast<int>(heap.size());
        vector<Result> results;
        results.reserve(M);

        while(!heap.empty()) {
            auto it = heap.top();
            heap.pop();
            const SubsetInfo& info = subsets[it.subset_idx];
            int n = static_cast<int>(info.indices.size());

            Result r;
            r.squared_error = it.err;
            r.indices = info.indices;
            r.weights = it.weights;
            results.push_back(move(r));
        }

        sort(results.begin(), results.end(), [](auto const& a, auto const& b) { return a.squared_error < b.squared_error; });
        return results;
    }

  private:
    vector<EigenColor> paints;
    const vector<int> comb_size;
    int K;

    struct SubsetInfo {
        vector<int> indices; // 部分集合の絵の具インデックス
    };
    vector<SubsetInfo> subsets;

    void prepare_subsets() {
        for(int sz : comb_size) {
            vector<int> comb(sz);
            function<void(int, int)> dfs = [&](int start, int depth) {
                if(depth == sz) {
                    SubsetInfo info;
                    info.indices = comb;
                    subsets.push_back(move(info));
                    return;
                }
                for(int x = start; x < K; x++) {
                    comb[depth] = x;
                    dfs(x + 1, depth + 1);
                }
            };
            dfs(0, 0);
        }
    }
};
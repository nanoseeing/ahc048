
#include <bits/stdc++.h>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>

#include "hpp/nnls.hpp"

using EigenColor = Eigen::Vector3d;

class ColorMixer {
  public:
    struct Result {
        double squared_error;   // 真の二乗誤差
        vector<int> indices;    // 部分集合の絵の具インデックス
        vector<double> weights; // 各絵の具の重み (合計 = 1)
        EigenColor mixed_color; // 混合後の色ベクトル
    };

    // paints_input: 任意本数の絵の具ベクトル (各要素は CMY の 3 次元)
    ColorMixer(const vector<EigenColor>& paints_input) : paints(paints_input) {
        K = static_cast<int>(paints.size());
        prepare_subsets();
    }

    // 目標色 t に対し、サイズ 2～4 の部分集合で NNLS を解き、誤差が小さい上位 topN 件を返す
    vector<Result> find_topN(const EigenColor& t, int topN) const {
        const double t_norm2 = t.squaredNorm();

        struct HeapItem {
            double err;
            int subset_idx;
            bool operator<(HeapItem const& o) const {
                return err < o.err;
            }
        };
        priority_queue<HeapItem> heap;

        int total = static_cast<int>(subsets.size());

        for(int si = 0; si < total; si++) {
            const SubsetInfo& info = subsets[si];
            int n = static_cast<int>(info.indices.size()); // 2,3,4 のいずれか

            // (1) 部分集合に対応する 3×n 行列 A_S を作る
            Eigen::Matrix<double, 3, Eigen::Dynamic> A(3, n);
            for(int j = 0; j < n; j++) {
                A.col(j) = paints[info.indices[j]];
            }

            // (2) NNLS を構築して解く
            Eigen::NNLS<Eigen::Matrix<double, 3, Eigen::Dynamic>> nnls_solver;
            nnls_solver.compute(A);
            nnls_solver.setTolerance(1e-12);
            nnls_solver.setMaxIterations(2 * n);
            Eigen::VectorXd w = nnls_solver.solve(t);
            if(nnls_solver.info() != Eigen::ComputationInfo::Success) {
                // 非負制約解が収束しなかった場合は一様分配しておく
                double uni = 1.0 / n;
                w = Eigen::VectorXd::Constant(n, uni);
            }

            // (3) w >= 0 かつ sum = 1 に正規化
            double sumw = w.sum();
            if(sumw <= 1e-12) {
                double uni = 1.0 / n;
                for(int i = 0; i < n; i++)
                    w(i) = uni;
                sumw = 1.0;
            } else {
                w /= sumw;
            }

            // (4) 混合後の色 c_hat = A * w
            EigenColor c_hat = A * w;

            // (5) 真の二乗誤差 E = || t - c_hat ||^2
            double E = (t - c_hat).squaredNorm();
            double true_err = 1e4 * E; // 1e4 を掛けたスコアを使う

            // (6) ヒープに突っ込む (topN を維持)
            if(static_cast<int>(heap.size()) < topN) {
                heap.push({true_err, si});
            } else if(true_err < heap.top().err) {
                heap.pop();
                heap.push({true_err, si});
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

            // A_S を再構築して NNLS をもう一度解く (重みを取り直す)
            Eigen::Matrix<double, 3, Eigen::Dynamic> A(3, n);
            for(int j = 0; j < n; j++) {
                A.col(j) = paints[info.indices[j]];
            }
            Eigen::NNLS<Eigen::Matrix<double, 3, Eigen::Dynamic>> nnls_solver;
            nnls_solver.compute(A);
            nnls_solver.setTolerance(1e-12);
            nnls_solver.setMaxIterations(2 * n);
            Eigen::VectorXd w = nnls_solver.solve(t);

            double sumw = w.sum();
            if(sumw <= 1e-12) {
                double uni = 1.0 / n;
                w = Eigen::VectorXd::Constant(n, uni);
            } else {
                w /= sumw;
            }

            EigenColor c_hat = A * w;

            Result r;
            r.squared_error = it.err;
            r.indices = info.indices;
            r.weights.resize(n);
            for(int i = 0; i < n; i++) {
                r.weights[i] = w(i);
            }
            r.mixed_color = c_hat;
            results.push_back(move(r));
        }

        sort(results.begin(), results.end(), [](auto const& a, auto const& b) { return a.squared_error < b.squared_error; });
        return results;
    }

  private:
    vector<EigenColor> paints;
    int K;

    struct SubsetInfo {
        vector<int> indices; // 部分集合の絵の具インデックス
    };
    vector<SubsetInfo> subsets;

    void prepare_subsets() {
        const vector<int> COMB_SIZES = {2, 3, 4};
        for(int sz : COMB_SIZES) {
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

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // -------------------------------------------------------------
    // サンプル：4 本の絵の具を用意し、ColorMixer を作成
    // -------------------------------------------------------------
    vector<EigenColor> paints = {EigenColor(0.10, 0.20, 0.30), EigenColor(0.80, 0.10, 0.10), EigenColor(0.25, 0.75, 0.25), EigenColor(0.60, 0.20, 0.20)};

    ColorMixer mixer(paints);

    // 目標色 t を指定
    EigenColor t;
    t << 0.33, 0.47, 0.20;

    // 上位 5 組み合わせを取得
    int topN = 5;
    auto results = mixer.find_topN(t, topN);

    cout << fixed << setprecision(6);
    cout << "Top " << topN << " combinations:\n";
    for(int i = 0; i < (int)results.size(); i++) {
        const auto& r = results[i];
        cout << "#" << (i + 1) << "  squared_error=" << r.squared_error << "  [";
        for(int j = 0; j < (int)r.indices.size(); j++) {
            cout << r.indices[j] << (j + 1 < (int)r.indices.size() ? ", " : "");
        }
        cout << "]\n";
        cout << "    weights: [";
        for(int j = 0; j < (int)r.weights.size(); j++) {
            cout << r.weights[j] << (j + 1 < (int)r.weights.size() ? ", " : "");
        }
        cout << "]\n";
        cout << "    mixed_color: (" << r.mixed_color[0] << ", " << r.mixed_color[1] << ", " << r.mixed_color[2] << ")\n";
    }

    return 0;
}

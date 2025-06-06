#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <vector>

#include "hpp/batch_inv.hpp"
#include "hpp/batch_pdb.hpp"
#include "hpp/comb.hpp"
#include "hpp/common.hpp"
#include "hpp/game.hpp"
#include "hpp/io.hpp"
#include "hpp/utils.hpp"

// 前節の batchSolveByPGD, batchSolveByInvClip, ProjectOntoSimplex をインクルード済みとする

// 例: Color を std::array<double,3> と定義しておく
using Color = std::array<double, 3>;

// きっかけ: paints という vector<Color>（全体の絵の具プール）がある。
// indices: サブセットとして選んだ絵の具のインデックス vector<int> indices;
// targets1024: std::vector<Color> 型で 1024 色の目標色が格納されているとする。

void exampleUsage(const std::vector<Color>& paints,
                  const std::vector<int>& indices,      // サブセットのインデックス (size = n)
                  const std::vector<Color>& targets1024 // 1024 色の目標色 (size = 1024)
) {
    int n = indices.size();          // サブセットのサイズ
    int M = (int)targets1024.size(); // 1024 のはず

    // 1) サブセット行列 A_s を作る (3 × n)
    Eigen::Matrix<double, 3, Eigen::Dynamic> A_s(3, n);
    for(int j = 0; j < n; ++j) {
        const auto& c = paints[indices[j]];
        A_s(0, j) = c[0];
        A_s(1, j) = c[1];
        A_s(2, j) = c[2];
    }

    // 2) 目標色行列 T_all を作る (3 × M)
    Eigen::Matrix<double, 3, Eigen::Dynamic> T_all(3, M);
    for(int k = 0; k < M; ++k) {
        const auto& t = targets1024[k];
        T_all(0, k) = t[0];
        T_all(1, k) = t[1];
        T_all(2, k) = t[2];
    }

    // --- (A) PGD 版を使う ---
    // X_init を特に与えなければ自動で (1/n,...,1/n) に初期化される。
    Eigen::MatrixXd X_init; // 空のまま渡すと自動初期化される

    // BatchPGDResult pgdResult = batchSolveByPGD(A_s, T_all, X_init,
    //                                            /*tol=*/1e-6,
    //                                            /*max_iter=*/100,
    //                                            /*is_alpha_max_fixed=*/true);

    // for(int k = 0; k < M; ++k) {
    //     cerr << k << "::" << pgdResult.err[k] * 1e4 << "\n";
    // }
    // cerr << endl;

    // 結果が pgdResult.X_final (n×M) に格納されている。
    // pgdResult.err[k] に色 k の誤差値が入っている。

    // たとえば色 k の重みベクトルは:
    //   Eigen::VectorXd w_k = pgdResult.X_final.col(k);

    // // --- (B) 逆行列＋クリップ 版を使う ---
    BatchInvClipResult invResult = batchSolveByInvClip(A_s, T_all,
                                                       /*eps=*/1e-9);
    for(int k = 0; k < M; ++k) {
        cerr << k << "::" << invResult.err[k] * 1e4 << "\n";
    }
    cerr << endl;

    // 結果が invResult.W_clip (n×M) に格納されている。
    // invResult.err[k] に色 k の誤差値が入っている。

    // たとえば色 k の重みベクトルは:
    //   Eigen::VectorXd w_k_clip = invResult.W_clip.col(k);
}

void solve() {
    Input input = parse_input();
    ColorMixer mixer(input.own);

    TimeKeeper timer(100.0);
    exampleUsage(input.own, {0, 1, 2, 3, 4}, input.target);

    // const int SEARCH_SIZE = 10;

    // for(int h : range(input.H)) {
    //     Color t = input.target[h];
    //     vector<double> errs;
    //     for(int k : range(2, min(5, input.K + 1))) {
    //         auto results = mixer.find_topN(t, SEARCH_SIZE);
    //         // auto results = mixer.solve_nnls(t, k, SEARCH_SIZE);
    //         auto best_ret = results[0];
    //         double w_sum = accumulate(ALL(best_ret.weights), 0.0);
    //         double e = best_ret.squared_error * 1e4;
    //         errs.push_back(e);
    //     }
    //     // cpp_dump(h, errs);
    // }

    cpp_dump(timer.getElapsedTime());
}

int main() {
    solve();
    return 0;
}
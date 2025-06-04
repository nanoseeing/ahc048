// QPSolverExample.cpp
//
// 「非負かつ合計=1」で最小二乗を解く例。
// QuadProg++ を使って QP を解く。
//
//   minimize (1/2) w^T G w + g0^T w
//   subject to   CE^T w = ce0    （CE: 合計=1 のイコール制約）
//                CI^T w >= ci0   （CI: 非負制約）
//
//--------------------------------------------------------------------------------

#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

// 1. QuadProg++ のヘッダー（ダウンロードしてプロジェクトに置いておくこと）
//    https://github.com/liuq/QuadProgpp の中の QuadProg++.hh を使います。
#include "hpp/ex/QuadProg++.hh"

using namespace std;

//--------------------------------------------------------------------------------
// solveConvexComboQP
//   与えられた「塗料ベクトル A (3×k)」と「目標色 t (3×1)」に対して、
//   以下を解く。
//       minimize ||A w - t||^2
//       subject to w >= 0,  sum_i w_i = 1.
//
//   返り値：
//     - 成功したら weights (size=k) に解を格納して true を返す。
//     - 失敗（不適定行列・収束失敗など）なら false を返す。
//--------------------------------------------------------------------------------
bool solveConvexComboQP(const Eigen::Matrix<double, 3, Eigen::Dynamic>& A, // 3×k
                        const Eigen::Vector3d& t,                          // 3×1
                        Eigen::VectorXd& weights                           // 出力ベクトル (k×1)
) {
    const int k = static_cast<int>(A.cols());
    //--------------------------------------------------------------------------------
    // (1) QP のパラメータ G, g0 を作成する
    //     もともとの目的関数 ||A w - t||^2 を拡張：
    //       w^T(A^T A) w - 2 (A^T t)^T w + const
    //     なので
    //       G = 2 (A^T A),   g0 = -2 (A^T t)
    //--------------------------------------------------------------------------------
    Eigen::MatrixXd ATA = A.transpose() * A; // k×k
    Eigen::VectorXd ATt = A.transpose() * t; // k×1

    Eigen::MatrixXd G = 2.0 * ATA;   // (k×k) 正定行列（半正定?）
    Eigen::VectorXd g0 = -2.0 * ATt; // (k×1)

    //--------------------------------------------------------------------------------
    // (2) 等式制約 (sum_i w_i = 1) を CE^T * w = ce0 の形にする
    //     QuadProg++ では、
    //       CE: n×p 行列（各列がイコール制約の共通ベクトル）
    //       ce0: p×1 ベクトル（各等式制約の右辺）
    //     CE^T w = ce0 を満たすとき w_i の合計=1 を課すには、
    //       CE が「全要素 1 の k×1 ベクトル」, ce0 = [1] となる。
    //--------------------------------------------------------------------------------
    const int p = 1; // イコール制約の数
    QuadProgPP::Matrix<double> CE(k, p);
    QuadProgPP::Vector<double> ce0(p);

    // CE の唯一の列をすべて 1 にする
    for(int i = 0; i < k; i++) {
        CE[i][0] = 1.0;
    }
    // CE^T w = 1 を課す
    ce0[0] = 1.0;

    //--------------------------------------------------------------------------------
    // (3) 不等式制約 (w_i >= 0) を CI^T * w >= ci0 の形にする
    //     QuadProg++ では、
    //       CI: n×m 行列（各列が不等式制約の係数ベクトル）
    //       ci0: m×1 ベクトル（各不等式制約の右辺）
    //     「w_i >= 0」を CI^T w >= ci0 に変形すると CI の各列が単位ベクトル e_i 、
    //     ci0[i] = 0 となる。
    //
    //     つまり
    //       w1 >= 0  →  [1, 0, 0, …, 0]^T w >= 0
    //       w2 >= 0  →  [0, 1, 0, …, 0]^T w >= 0
    //        …
    //       wk >= 0  →  [0, 0, …, 1]^T w >= 0
    //--------------------------------------------------------------------------------
    const int m = k; // 不等式制約の数（w_i >= 0 が k 本）
    QuadProgPP::Matrix<double> CI(k, m);
    QuadProgPP::Vector<double> ci0(m);

    // CI の各列を単位ベクトルに、ci0 はすべて 0
    for(int j = 0; j < m; j++) {
        for(int i = 0; i < k; i++) {
            CI[i][j] = (i == j ? 1.0 : 0.0);
        }
        ci0[j] = 0.0;
    }

    //--------------------------------------------------------------------------------
    // (4) QuadProg++ 用のデータ型に変換する
    //--------------------------------------------------------------------------------
    // QuadProg++ 標準の行列/ベクトル型は「Matrix<T>」「Vector<T>」クラスです。
    // Eigen の G (k×k), g0 (k×1) を QuadProg++ の形にコピーします。
    QuadProgPP::Matrix<double> G_qp(k, k);
    QuadProgPP::Vector<double> g0_qp(k);

    for(int i = 0; i < k; i++) {
        g0_qp[i] = g0(i);
        for(int j = 0; j < k; j++) {
            G_qp[i][j] = G(i, j);
        }
    }

    //--------------------------------------------------------------------------------
    // (5) QP を解く
    //       solve_quadprog( G, g0, CE, ce0, CI, ci0, w_qp );
    //     戻り値＝true なら成功。失敗なら false。
    //--------------------------------------------------------------------------------
    QuadProgPP::Vector<double> w_qp(k);              // 解を格納するベクトル (k×1)
    bool success = QuadProgPP::solve_quadprog(G_qp,  // G（k×k）
                                              g0_qp, // g0（k×1）
                                              CE,    // CE（k×p） p=1
                                              ce0,   // ce0（p×1）
                                              CI,    // CI（k×m） m=k
                                              ci0,   // ci0（m×1）
                                              w_qp   // <- ここに解が入る
    );

    if(!success) {
        // QP 自体が infeasible / 数値的に失敗
        cerr << "[QP solver] infeasible or numeric failure\n";
        return false;
    }

    //-----------------------
    // (6) 求まった解を呼び出し元の Eigen::Vector にコピー
    //-----------------------
    weights.resize(k);
    for(int i = 0; i < k; i++) {
        weights(i) = w_qp[i];
    }

    return true;
}

//--------------------------------------------------------------------------------
// main()
//   例として、「3本の絵の具」を用意し、
//   その中から k=3 を選んでターゲット色 t を再現するケースを示す。
//   （もちろん paint の本数や k は自由に変えてOK）
//--------------------------------------------------------------------------------
int main() {
    // --- (A) まず手持ちの K 本の絵の具ベクトルを用意する ---
    //     ここでは K = 4 として、各要素は CMY（3次元ベクトル）を想定
    vector<Eigen::Vector3d> paint_list;
    paint_list.emplace_back(1.0, 0.0, 0.0); // 塗料 #0 : 明度が強いシアン
    paint_list.emplace_back(0.0, 1.0, 0.0); // 塗料 #1 : マゼンタ強め
    paint_list.emplace_back(0.0, 0.0, 1.0); // 塗料 #2 : イエロー強め
    paint_list.emplace_back(0.5, 0.5, 0.0); // 塗料 #3 : シアン/マゼンタ混合

    // --- (B) 部分集合サイズ k を決める（今回は k=3 の例） ---
    const int k = 3;

    // 塗料インデックス集合から「0,1,2 を選ぶ」と仮定
    vector<int> subset_indices = {0, 1, 2};
    // もし全探索で回すなら for(...) で indices をすべて変えていけば良い

    // (C) 部分集合に対応する行列 A (3×k) を作る
    Eigen::Matrix<double, 3, Eigen::Dynamic> A(3, k);
    for(int j = 0; j < k; j++) {
        A.col(j) = paint_list[subset_indices[j]];
    }

    // --- (D) ターゲット色 t を用意する ---
    //     ここでは明示的に「#0,#1,#2 を等量(1/3ずつ)で混ぜた色」をターゲットとする。
    Eigen::Vector3d t = (paint_list[0] + paint_list[1] + paint_list[2]) / 3.0;

    // --- (E) QP を解いて重み w を求める ---
    Eigen::VectorXd w; // (k×1) 解ベクトル
    bool ok = solveConvexComboQP(A, t, w);
    if(!ok) {
        cout << "Failed to solve QP\n";
        return 1;
    }

    // --- (F) 結果を表示 ---
    cout << "Optimized weights (sum=" << w.sum() << "):\n";
    for(int i = 0; i < k; i++) {
        cout << "  w[" << subset_indices[i] << "] = " << w(i) << "\n";
    }
    // 混合結果も確認してみる
    Eigen::Vector3d c_hat = A * w;
    cout << "Reconstructed color c_hat = [" << c_hat(0) << ", " << c_hat(1) << ", " << c_hat(2) << "]\n";
    cout << "Target color t = [" << t(0) << ", " << t(1) << ", " << t(2) << "]\n";
    cout << "Error ||c_hat - t||^2 = " << (c_hat - t).squaredNorm() << "\n";

    return 0;
}

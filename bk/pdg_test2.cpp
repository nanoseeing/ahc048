using namespace std;

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm> // std::sort, std::iota
#include <functional>
#include <iostream>
#include <limits>
#include <numeric> // std::iota
#include <vector>

#include "hpp/common.hpp"
#include "hpp/io.hpp"
#include "hpp/utils.hpp"

// -----------------------------------------------------------------------------
// ProjectOntoSimplex()
//   ベクトル v (size=k) を、「w >= 0, sum(w)=1 の単純体」にユークリッド距離で
//   射影した結果を返す。
//   参考アルゴリズム：
//     [Michelot, “A finite algorithm for finding the projection of a point...,” 1986]
//   あるいは [Wang & Carreira-Perpiñán, “Projection onto the probability simplex”]
// -----------------------------------------------------------------------------
Eigen::VectorXd ProjectOntoSimplex(const Eigen::VectorXd& v) {
    const int k = static_cast<int>(v.size());
    // 1) v の要素を降順ソートするが、もとのインデックスにもアクセスできるように
    //    pair<値, 元のインデックス> をつくってソートするか、単に値のみソートする。
    //    ここでは「値のみソートして閾値を計算し、最後に元の順序に書き戻す」方式を取る。
    std::vector<double> u(k);
    for(int i = 0; i < k; i++)
        u[i] = v[i];
    std::sort(u.begin(), u.end(), std::greater<double>()); // 降順ソート

    // 2) 累積和をとりながら rho を見つける
    std::vector<double> cumsum(k, 0.0);
    cumsum[0] = u[0];
    for(int i = 1; i < k; i++) {
        cumsum[i] = cumsum[i - 1] + u[i];
    }

    int rho = -1;
    double theta = 0;
    for(int j = 0; j < k; j++) {
        // 注意： j は 0-based なので “j+1” が人間読みの要素数
        double t = (cumsum[j] - 1.0) / (j + 1);
        if(u[j] - t > 0) {
            rho = j;   // 0-based index のまま保持
            theta = t; // 候補の θ 値
        }
    }
    // 3) theta は「最後に更新された (cumsum[rho]-1)/(rho+1)」
    //    もし rho=-1 だったら、全要素が u[j] - t <= 0 ということだが、
    //    でも v がすべて負になるケースは稀なので rho>=0 と仮定する。
    //    それでも念のため rho<0 の場合は一様分布にしておく：
    if(rho < 0) {
        // v 内のすべての要素が小さすぎて rho=−1 になったときは、
        // 単純に (1/k,...,1/k) を返す
        return Eigen::VectorXd::Constant(k, 1.0 / k);
    }

    // 4) 射影ベクトルを作る： w_i = max(v_i - theta, 0)
    Eigen::VectorXd w(k);
    for(int i = 0; i < k; i++) {
        w[i] = std::max(v[i] - theta, 0.0);
    }
    return w;
}

// -----------------------------------------------------------------------------
// solveConvexComboByPGD()
//   A: 3×k 行列 (各列が「選択した k 本の塗料ベクトル」)
//   t: 3×1 目標色ベクトル
//   maxIter: 最大反復回数
//   alpha: ステップサイズ（固定ステップとして使う。小さければ確実だが要チューニング）
//   tol: 収束判定閾値（解の変化量や勾配ノルムで判定できる）
//
//   戻り値 w は k×1 で、非負かつ sum(w)=1 を満たす最適想定値。
//   もし収束しなければ最後の反復値を返す。
// -----------------------------------------------------------------------------
Eigen::VectorXd solveConvexComboByPGD(const Eigen::Matrix<double, 3, Eigen::Dynamic>& A, const Eigen::Vector3d& t, int maxIter = 1000, double alpha = 1e-12,
                                      double tol = 1e-8) {
    const int k = static_cast<int>(A.cols());
    // (1) 初期化：単純体の中心 (1/k,1/k,...,1/k)
    Eigen::VectorXd w = Eigen::VectorXd::Constant(k, 1.0 / k);

    // 反復に必要な一時変数
    Eigen::VectorXd grad(k);
    Eigen::VectorXd Aw_minus_t(3);

    for(int iter = 0; iter < maxIter; iter++) {
        // (2) 勾配を計算：∇f(w) = 2 Aᵀ (A w - t)
        //     まず A w - t を求める (3×1)
        Aw_minus_t = A * w - t;
        //     次に勾配（k×1）
        grad = 2.0 * (A.transpose() * Aw_minus_t);

        // (3) 更新候補： w_tmp = w - alpha * grad
        Eigen::VectorXd w_tmp = w - alpha * grad;

        // (4) 単純体への射影
        Eigen::VectorXd w_next = ProjectOntoSimplex(w_tmp);

        // (5) 収束判定：||w_next - w|| が tol 未満なら break
        double diff = (w_next - w).norm();
        w = std::move(w_next);
        if(diff < tol) {
            // 十分に収束したと判断
            break;
        }
    }

    return w;
}

// 部分集合を DFS で列挙して PGD をかける関数
void searchAllSubsets(const vector<Eigen::Vector3d>& paint_list, // K 本の塗料
                      const Eigen::Vector3d& t,                  // 目標色
                      int k_subset,                              // 部分集合サイズ k
                      vector<int>& best_indices,                 // 最良組み合わせを返すバッファ
                      Eigen::VectorXd& best_weights,             // そのときの重みを返すバッファ
                      double& best_error                         // 最良誤差を返すバッファ
) {
    const int K = static_cast<int>(paint_list.size());
    vector<int> comb(k_subset, 0);

    // 再帰 DFS
    function<void(int, int)> dfs = [&](int start, int depth) {
        if(depth == k_subset) {
            // 部分集合 comb が決まった → 3×k_subset 行列 A を作成
            Eigen::Matrix<double, 3, Eigen::Dynamic> A(3, k_subset);
            for(int j = 0; j < k_subset; j++) {
                A.col(j) = paint_list[comb[j]];
            }
            // PGD で解を求める
            Eigen::VectorXd w = solveConvexComboByPGD(A, t, 500, 1e-2, 1e-8);
            // 再構成誤差を計算
            Eigen::Vector3d c_hat = A * w;
            double err = (t - c_hat).squaredNorm();
            if(err < best_error) {
                best_error = err;
                best_indices = comb; // 部分集合インデックスを保存
                best_weights = w;    // 重みベクトルを保存
            }
            return;
        }
        for(int i = start; i < K; i++) {
            comb[depth] = i;
            dfs(i + 1, depth + 1);
        }
    };

    dfs(0, 0);
}

int main() {
    Input input = parse_input();
    vector<Eigen::Vector3d> paint_list;
    for(const auto& col : input.own) {
        Eigen::Vector3d c(col[0], col[1], col[2]);
        paint_list.push_back(c);
    }

    int K = static_cast<int>(paint_list.size());

    for(int h : range(input.H)) {
        auto t_color = input.target[h];
        Eigen::Vector3d t = Eigen::Vector3d(t_color[0], t_color[1], t_color[2]);
        const int k_subset = 4;
        vector<int> best_indices;
        Eigen::VectorXd best_weights;
        double best_error = numeric_limits<double>::infinity();
        searchAllSubsets(paint_list, t, k_subset, best_indices, best_weights, best_error);
        cpp_dump(best_error * 1e4, best_weights); //, best_indices, best_weights);
    }

    // // --------- (E) 結果を表示 ------------
    // cout << "Best combination (size=" << k_subset << "): ";
    // for(int idx : best_indices)
    //     cout << idx << " ";
    // cout << "\nWeights: ";
    // for(int i = 0; i < k_subset; i++) {
    //     cout << best_weights[i] << " ";
    // }
    // cout << "\nReconstruction error: " << best_error << "\n";

    // // 再構成色も確認しておく
    // Eigen::Matrix<double, 3, Eigen::Dynamic> A(3, k_subset);
    // for(int j = 0; j < k_subset; j++) {
    //     A.col(j) = paint_list[best_indices[j]];
    // }
    // Eigen::Vector3d c_hat = A * best_weights;
    // cout << "Reconstructed color: [" << c_hat[0] << ", " << c_hat[1] << ", " << c_hat[2] << "]\n";
    // cout << "Target color:        [" << t[0] << ", " << t[1] << ", " << t[2] << "]\n";

    return 0;
}

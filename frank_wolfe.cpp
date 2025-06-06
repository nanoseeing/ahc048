#include <bits/stdc++.h>

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
bool frank_wolfe_simplex_ls(const Eigen::MatrixXd &A, const Eigen::VectorXd &b,
                            Eigen::VectorXd &x, // 初期値と解 (size n)
                            double tol = 1e-10, int max_iter = 10000) {
    const int n = x.size();
    const Eigen::MatrixXd ATA = A.transpose() * A;
    const Eigen::VectorXd ATb = A.transpose() * b;

    for(int iter = 0; iter < max_iter; ++iter) {
        // 勾配
        Eigen::VectorXd g = ATA * x - ATb;

        // 最も小さい勾配成分のインデックス
        int s_idx;
        g.minCoeff(&s_idx);

        // s: 標準基底ベクトル（その成分だけ1）
        Eigen::VectorXd s = Eigen::VectorXd::Zero(n);
        s[s_idx] = 1.0;

        // ステップ方向
        Eigen::VectorXd d = s - x;

        // 最適ステップサイズγを解析的に計算
        Eigen::VectorXd Ad = A * d;
        Eigen::VectorXd Ax = A * x;
        double num = (Ax - b).dot(Ad);
        double denom = Ad.squaredNorm();
        double gamma = (denom == 0.0) ? 0.0 : std::max(0.0, std::min(1.0, -num / denom));

        // x を更新
        x += gamma * d;

        // 収束判定（KKT: d^T grad = g[s_idx] - g^T x = 最小勾配と現在xの勾配差）
        double gap = g.dot(x) - g[s_idx];
        if(fabs(gap) < tol) {
            return true;
        }
    }
    return false;
}

int main() {
    int K = 20, m = 3;
    Eigen::MatrixXd A(m, K);
    Eigen::VectorXd b(m);

    // --- ここで A, b を設定 ---
    std::mt19937_64 rng(123456);
    std::uniform_real_distribution<double> distA(0.0, 1.0);
    for(int i = 0; i < m; ++i)
        b[i] = distA(rng);
    for(int j = 0; j < K; ++j)
        for(int i = 0; i < m; ++i)
            A(i, j) = distA(rng);

    // 単体上の初期値（均等分布）
    Eigen::VectorXd x = Eigen::VectorXd::Constant(K, 1.0 / K);

    bool ok = frank_wolfe_simplex_ls(A, b, x, 1e-12, 50000);

    std::cout << "Converged: " << ok << "\n";
    std::cout << "sum(x): " << x.sum() << "\n";
    std::cout << "||Ax-b||: " << (A * x - b).norm() << "\n";
    return 0;
}

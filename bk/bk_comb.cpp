#include <bits/stdc++.h>
using namespace std;

/*
  20本の絵の具ベクトル（CMY成分）を適当な数値で埋めています。
  実際にはお手持ちのデータに置き換えてください。
*/
static double paint[20][3] = {{0.10, 0.20, 0.30}, {0.15, 0.05, 0.50}, {0.40, 0.10, 0.10}, {0.50, 0.50, 0.00},
                              {0.25, 0.75, 0.25}, {0.80, 0.10, 0.10}, {0.90, 0.05, 0.05}, {0.60, 0.20, 0.20},
                              {0.30, 0.60, 0.10}, {0.20, 0.30, 0.70}, {0.05, 0.85, 0.10}, {0.45, 0.45, 0.10},
                              {0.70, 0.20, 0.10}, {0.10, 0.40, 0.50}, {0.55, 0.15, 0.30}, {0.35, 0.35, 0.30},
                              {0.65, 0.25, 0.10}, {0.80, 0.10, 0.10}, {0.10, 0.10, 0.80}, {0.50, 0.25, 0.25}};

/*
  部分集合の情報を保持する構造体
*/
struct SubsetInfo {
    int size;            // 部分集合のサイズ (2,3,4)
    array<int, 4> idx;   // 部分集合に含まれる絵の具のインデックス
    double pseudo[4][3]; // 擬似逆行列 (size × 3)。w_ls = pseudo × t
    double Gram[4][4];   // Gram 行列 = A_S^T * A_S (size × size 部分のみ)
};

static vector<SubsetInfo> all_subsets; // 全 6175 通りの部分集合

// Gauss-Jordan で size×size 行列を逆行列にする (size=2,3,4) 固定サイズ対応
void invertMatrix(const double G_in[4][4], double invG_out[4][4], int size) {
    double tmp[4][8];
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            tmp[i][j] = G_in[i][j];
        }
        for(int j = 0; j < size; j++) {
            tmp[i][size + j] = (i == j ? 1.0 : 0.0);
        }
    }
    for(int i = 0; i < size; i++) {
        // ピボット選択 (絶対値最大の行を i 行目と交換)
        int pivot = i;
        for(int r = i + 1; r < size; r++) {
            if(fabs(tmp[r][i]) > fabs(tmp[pivot][i])) {
                pivot = r;
            }
        }
        if(pivot != i) {
            for(int c = 0; c < 2 * size; c++) {
                swap(tmp[i][c], tmp[pivot][c]);
            }
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
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            invG_out[i][j] = tmp[i][size + j];
        }
    }
}

// 事前準備：サイズ2～4の全組み合わせ6175通りを列挙し、pseudoとGramを計算して保存
void prepare_all_subsets() {
    for(int sz = 2; sz <= 4; sz++) {
        vector<int> comb(sz);
        function<void(int, int)> dfs = [&](int start, int depth) {
            if(depth == sz) {
                SubsetInfo info;
                info.size = sz;
                for(int i = 0; i < sz; i++) {
                    info.idx[i] = comb[i];
                }
                // Gram = A_S^T * A_S (sz×sz)
                for(int i = 0; i < sz; i++) {
                    for(int j = 0; j < sz; j++) {
                        double dot = 0.0;
                        for(int d = 0; d < 3; d++) {
                            dot += paint[comb[i]][d] * paint[comb[j]][d];
                        }
                        info.Gram[i][j] = dot;
                    }
                }
                // Gram の逆行列を求める
                double invG[4][4] = {{0}};
                invertMatrix(info.Gram, invG, sz);
                // 擬似逆行列 pseudo = invG × A_S^T (sz×3)
                for(int i = 0; i < sz; i++) {
                    for(int d = 0; d < 3; d++) {
                        double sum = 0.0;
                        for(int j = 0; j < sz; j++) {
                            sum += invG[i][j] * paint[comb[j]][d];
                        }
                        info.pseudo[i][d] = sum;
                    }
                }
                // 余白をゼロ埋め
                for(int i = sz; i < 4; i++) {
                    for(int d = 0; d < 3; d++) {
                        info.pseudo[i][d] = 0.0;
                    }
                }
                for(int i = 0; i < 4; i++) {
                    for(int j = sz; j < 4; j++) {
                        info.Gram[i][j] = info.Gram[j][i] = 0.0;
                    }
                }
                all_subsets.push_back(info);
                return;
            }
            for(int x = start; x < 20; x++) {
                comb[depth] = x;
                dfs(x + 1, depth + 1);
            }
        };
        dfs(0, 0);
    }
    // all_subsets.size() は 6175 になる
}

// 上位 TOPN 件を「(二乗誤差, インデックス配列, 重み配列, 混合色配列)」のタプルで返す
vector<tuple<double, array<int, 4>, array<double, 4>, array<double, 3>>> find_topN(const double t[3], int TOPN) {
    struct HeapItem {
        double err;      // 本当の二乗誤差
        int idx_in_list; // all_subsets のインデックス
        bool operator<(HeapItem const &o) const {
            return err < o.err; // priority_queue で最大ヒープ化（大きいerrが先頭）
        }
    };

    // まず t^T t を計算しておく
    double t_norm2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];

    priority_queue<HeapItem> heap;
    int Nall = (int)all_subsets.size(); // 6175

    for(int si = 0; si < Nall; si++) {
        const SubsetInfo &info = all_subsets[si];
        int n = info.size;

        // 1) 擬似逆行列 × t で「制約なし最小二乗解」w_ls を計算 (n次元)
        double w_ls[4];
        for(int i = 0; i < n; i++) {
            w_ls[i] = info.pseudo[i][0] * t[0] + info.pseudo[i][1] * t[1] + info.pseudo[i][2] * t[2];
        }
        for(int i = n; i < 4; i++) {
            w_ls[i] = 0.0;
        }

        // 2) クリッピング＋正規化
        double sum = 0.0;
        for(int i = 0; i < n; i++) {
            if(w_ls[i] < 0.0) w_ls[i] = 0.0;
            sum += w_ls[i];
        }
        if(sum <= 0.0) {
            for(int i = 0; i < n; i++) {
                w_ls[i] = 1.0 / n;
            }
        } else {
            for(int i = 0; i < n; i++) {
                w_ls[i] /= sum;
            }
        }

        // 3) b = A_S^T * t を計算 (n 次元)
        double b[4] = {0, 0, 0, 0};
        for(int i = 0; i < n; i++) {
            int pk = info.idx[i];
            b[i] = paint[pk][0] * t[0] + paint[pk][1] * t[1] + paint[pk][2] * t[2];
        }

        // 4) w^T G w と -2 b^T w を計算
        double wGw = 0.0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                wGw += w_ls[i] * info.Gram[i][j] * w_ls[j];
            }
        }
        double bTw = 0.0;
        for(int i = 0; i < n; i++) {
            bTw += b[i] * w_ls[i];
        }
        double eprime = wGw - 2.0 * bTw;

        // 5) 本当の二乗誤差を eprime + t_norm2 で計算
        double true_error = eprime + t_norm2;

        // 6) ヒープに入れて TOPN を維持
        if((int)heap.size() < TOPN) {
            heap.push({true_error, si});
        } else if(true_error < heap.top().err) {
            heap.pop();
            heap.push({true_error, si});
        }
    }

    // 7) ヒープに残った TOPN 件を vector に移して、「誤差昇順」に並べて返却。
    int M = (int)heap.size();
    vector<tuple<double, array<int, 4>, array<double, 4>, array<double, 3>>> result;
    result.reserve(M);

    while(!heap.empty()) {
        auto it = heap.top();
        heap.pop();
        const SubsetInfo &info = all_subsets[it.idx_in_list];
        int n = info.size;

        array<int, 4> Sidx = {-1, -1, -1, -1};
        array<double, 4> W = {0, 0, 0, 0};
        array<double, 3> C = {0, 0, 0}; // 混合後の色

        // 部分集合のインデックス
        for(int i = 0; i < n; i++) {
            Sidx[i] = info.idx[i];
        }
        // 重みを再計算 (pseudo×t → クリップ → 正規化) して W[i] に格納
        double w_ls[4];
        for(int i = 0; i < n; i++) {
            w_ls[i] = info.pseudo[i][0] * t[0] + info.pseudo[i][1] * t[1] + info.pseudo[i][2] * t[2];
        }
        for(int i = n; i < 4; i++) {
            w_ls[i] = 0.0;
        }
        double sum = 0.0;
        for(int i = 0; i < n; i++) {
            if(w_ls[i] < 0.0) w_ls[i] = 0.0;
            sum += w_ls[i];
        }
        if(sum <= 0.0) {
            for(int i = 0; i < n; i++) {
                w_ls[i] = 1.0 / n;
            }
        } else {
            for(int i = 0; i < n; i++) {
                w_ls[i] /= sum;
            }
        }
        for(int i = 0; i < n; i++) {
            W[i] = w_ls[i];
        }

        // 8) その重みで実際に混ぜた色ベクトル C を計算
        for(int d = 0; d < 3; d++) {
            double sumC = 0.0;
            for(int i = 0; i < n; i++) {
                int pk = info.idx[i];
                sumC += w_ls[i] * paint[pk][d];
            }
            C[d] = sumC;
        }

        result.emplace_back(it.err, Sidx, W, C);
    }

    // 誤差が小さい順にソート
    sort(result.begin(), result.end(), [&](auto const &a, auto const &b) { return get<0>(a) < get<0>(b); });
    return result;
}

int main() {
    // 事前準備 (一度だけ)
    prepare_all_subsets();

    // サンプル目標色 t を指定 (必要に応じて変更可)
    for(int i = 0; i < 1000; i++) {
        double t[3] = {0.33, 0.47, 0.20};

        // 上位 TOPN 件を取得 (例: 10 件)
        const int TOPN = 5000;
        auto topList = find_topN(t, TOPN);
    }

    // // 結果を出力
    // cout << "=== Target Color: (" << t[0] << ", " << t[1] << ", " << t[2] << ") ===\n";
    // cout << "Top " << TOPN << " combinations:\n";

    // for(int i = 0; i < (int)topList.size(); i++) {
    //     double err;
    //     array<int, 4> idxs;
    //     array<double, 4> wts;
    //     array<double, 3> mixed;
    //     tie(err, idxs, wts, mixed) = topList[i];

    //     // 部分集合サイズを idxs[] の -1 で判別
    //     int sz = 0;
    //     while(sz < 4 && idxs[sz] != -1)
    //         sz++;

    //     cout << "#" << setw(2) << i + 1 << "  squared_error=" << fixed << setprecision(6) << err << "  [";
    //     for(int j = 0; j < sz; j++) {
    //         cout << idxs[j];
    //         if(j < sz - 1) cout << ", ";
    //     }
    //     cout << "]\n";

    //     // 係数を表示
    //     cout << "    weights: [";
    //     for(int j = 0; j < sz; j++) {
    //         cout << fixed << setprecision(7) << wts[j];
    //         if(j < sz - 1) cout << ", ";
    //     }
    //     cout << "]\n";

    //     // 混合後の色を表示
    //     cout << "    mixed_color: (" << fixed << setprecision(4) << mixed[0] << ", " << mixed[1] << ", " << mixed[2]
    //          << ")\n";
    // }

    return 0;
}

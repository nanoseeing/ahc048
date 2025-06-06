#pragma once
#include "common.hpp"
#include "utils.hpp"

/************************************************************************************************
 * class ColorMixer
 *
 * - コンストラクタに任意個数の絵の具（CMY の 3 要素）を渡すと、
 *   サイズ 2〜4 のすべての組み合わせを事前計算しておきます (Gram 行列と擬似逆行列)。
 * - メソッド find_topN(target_color, N) を呼ぶと、二乗誤差が小さい上位 N 件を返します。
 * - Result 構造体で誤差・インデックス・重み・混合色をまとめて取得できます。
 ************************************************************************************************/

class ColorMixer {
  public:
    // 結果を格納する構造体
    struct Result {
        double squared_error;   // 本当の二乗誤差
        vector<int> indices;    // 組み合わせに使った絵の具のインデックス
        vector<double> weights; // 各絵の具の重み (size = indices.size())
        Color mixed_color;      // 混合後の色 (C, M, Y)
    };

    // 最大ヒープ (squared_error が大きいものが top に来る)
    struct HeapItem {
        double err;
        int subset_idx;
        bool operator<(HeapItem const& o) const {
            return err < o.err;
        }
    };

    // コンストラクタ：任意の数の絵の具 (CMY) を渡す
    ColorMixer(const vector<Color>& paints_input, const vector<int>& _comb_size) : paints(paints_input), comb_size(_comb_size) {
        K = paints.size();
        prepare_subsets();
    }

    tuple<vector<double>, double, Color> solve_nnls_for_indices(const vector<int>& indices, const Color& t) {
        SubsetInfo info;
        if(subset_map.find(indices) != subset_map.end()) {
            // 事前に計算済みの部分集合情報を取得
            info = subset_map.at(indices);
        } else {
            // 新しい組み合わせなので、計算して保存
            info = create_info(indices);
            subset_map[indices] = info;
        }

        int n = indices.size();
        double t_norm2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];

        // 1) 擬似逆行列 × t で制約なし最小二乗解を得る
        vector<double> w_ls(n, 0.0);
        for(int i = 0; i < n; i++) {
            // pseudo はサイズ n×3 の行列
            w_ls[i] = info.pseudo[i][0] * t[0] + info.pseudo[i][1] * t[1] + info.pseudo[i][2] * t[2];
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
            int pk = info.indices[i];
            b[i] = paints[pk][0] * t[0] + paints[pk][1] * t[1] + paints[pk][2] * t[2];
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

        // 5) 本当の二乗誤差 = eprime + t_norm2
        double true_err = eprime + t_norm2;

        // 混合後の色を計算
        Color C = {0.0, 0.0, 0.0};
        for(int d = 0; d < 3; d++) {
            double sumC = 0.0;
            for(int i = 0; i < n; i++) {
                int pk = info.indices[i];
                sumC += w_ls[i] * paints[pk][d];
            }
            C[d] = sumC;
        }

        return {w_ls, true_err, C};
    }

    // 目標色 t に対して、上位 topN 件を返す
    vector<Result> find_topN(const Color& t, int topN) const {
        // まず t^T t を計算しておく
        double t_norm2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];

        priority_queue<HeapItem> heap;

        int total = subsets.size(); // 事前に列挙した全組み合わせ数

        // 全組み合わせを走査
        for(int si = 0; si < total; si++) {
            const SubsetInfo& info = subsets[si];
            int n = info.size;

            // 1) 擬似逆行列 × t で制約なし最小二乗解を得る
            vector<double> w_ls(n, 0.0);
            for(int i = 0; i < n; i++) {
                // pseudo はサイズ n×3 の行列
                w_ls[i] = info.pseudo[i][0] * t[0] + info.pseudo[i][1] * t[1] + info.pseudo[i][2] * t[2];
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
                int pk = info.indices[i];
                b[i] = paints[pk][0] * t[0] + paints[pk][1] * t[1] + paints[pk][2] * t[2];
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

            // 5) 本当の二乗誤差 = eprime + t_norm2
            double true_err = eprime + t_norm2;

            // 6) ヒープに突っ込む
            if((int)heap.size() < topN) {
                heap.push({true_err, si});
            } else if(true_err < heap.top().err) {
                heap.pop();
                heap.push({true_err, si});
            }
        }

        // 7) ヒープに残った上位 topN 件を取り出し、誤差昇順にソートして返却
        int M = heap.size();
        vector<Result> results;
        results.reserve(M);

        set<vector<int>> unique_indices;

        while(!heap.empty()) {
            auto it = heap.top();
            heap.pop();
            const SubsetInfo& info = subsets[it.subset_idx];
            int n = info.size;

            // 部分集合のインデックスをコピー
            vector<int> idxs = info.indices;

            // 重みを改めて計算して W に入れる
            vector<double> W(n, 0.0);
            {
                vector<double> w_tmp(n, 0.0);
                for(int i = 0; i < n; i++) {
                    w_tmp[i] = info.pseudo[i][0] * t[0] + info.pseudo[i][1] * t[1] + info.pseudo[i][2] * t[2];
                }
                double sum2 = 0.0;
                for(int i = 0; i < n; i++) {
                    if(w_tmp[i] < 0.0) w_tmp[i] = 0.0;
                    sum2 += w_tmp[i];
                }
                if(sum2 <= 0.0) {
                    double uni = 1.0 / n;
                    for(int i = 0; i < n; i++) {
                        w_tmp[i] = uni;
                    }
                } else {
                    for(int i = 0; i < n; i++) {
                        w_tmp[i] /= sum2;
                    }
                }
                for(int i = 0; i < n; i++) {
                    W[i] = w_tmp[i];
                }
            }

            // 混合後の色を計算
            Color C = {0.0, 0.0, 0.0};
            for(int d = 0; d < 3; d++) {
                double sumC = 0.0;
                for(int i = 0; i < n; i++) {
                    int pk = info.indices[i];
                    sumC += W[i] * paints[pk][d];
                }
                C[d] = sumC;
            }

            vector<int> new_idxs = idxs;
            vector<double> new_w = W;

            // // 重みが 0 のものを除外
            // vector<int> new_idxs;
            // vector<double> new_w;
            // for(int i = 0; i < n; i++) {
            //     if(W[i] > 0.0) {
            //         new_idxs.push_back(idxs[i]);
            //         new_w.push_back(W[i]);
            //     }
            // }
            // // ユニークなインデックスの組み合わせをチェック
            // if(unique_indices.contains(new_idxs)) {
            //     continue;
            // } else {
            //     unique_indices.insert(new_idxs);
            // }

            // インデックスと重みを降順にソート
            auto temp_idxs = make_sorted_indices(new_w, true);
            reorder_vector(new_idxs, temp_idxs);
            reorder_vector(new_w, temp_idxs);

            // 結果を格納
            Result r;
            r.squared_error = it.err;
            r.indices = move(new_idxs);
            r.weights = move(new_w);
            r.mixed_color = C;
            results.push_back(r);
        }

        // 誤差が小さい順にソート
        sort(results.begin(), results.end(), [&](Result const& a, Result const& b) { return a.squared_error < b.squared_error; });
        return results;
    }

  private:
    // 絵の具データ
    vector<Color> paints;
    const vector<int> comb_size;
    int K; // 絵の具の数

    // 各組み合わせを表す情報
    struct SubsetInfo {
        int size;                      // 組み合わせサイズ
        vector<int> indices;           // 絵の具インデックス (size 要素)
        vector<vector<double>> Gram;   // Gram 行列: size×size
        vector<vector<double>> pseudo; // 擬似逆行列: size×3
    };

    vector<SubsetInfo> subsets;              // 事前に構築する全組み合わせ情報
    map<vector<int>, SubsetInfo> subset_map; // インデックスから SubsetInfo へのマップ

    // size×size の小行列を逆行列にする (size <= 4 を想定)
    void invertMatrix(const vector<vector<double>>& G, vector<vector<double>>& invG, int size) const {
        // tmp は size × (2*size) の拡大行列 [G | I]
        vector<vector<double>> tmp(size, vector<double>(2 * size, 0.0));
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                tmp[i][j] = G[i][j];
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

    SubsetInfo create_info(const vector<int>& indices) {
        SubsetInfo info;
        info.size = static_cast<int>(indices.size());
        info.indices = indices;

        int sz = info.size;

        // A_S^T * A_S = Gram (sz×sz)
        info.Gram.assign(sz, vector<double>(sz, 0.0));
        for(int i = 0; i < sz; i++) {
            for(int j = 0; j < sz; j++) {
                double dot = 0.0;
                for(int d = 0; d < 3; d++) {
                    dot += paints[indices[i]][d] * paints[indices[j]][d];
                }
                info.Gram[i][j] = dot;
            }
        }

        // Gram の逆行列 invG を計算
        vector<vector<double>> invG;
        invertMatrix(info.Gram, invG, sz);

        // 擬似逆行列 = invG × A_S^T (sz×3)
        info.pseudo.assign(sz, vector<double>(3, 0.0));
        for(int i = 0; i < sz; i++) {
            for(int d = 0; d < 3; d++) {
                double sum = 0.0;
                for(int j = 0; j < sz; j++) {
                    sum += invG[i][j] * paints[indices[j]][d];
                }
                info.pseudo[i][d] = sum;
            }
        }

        return info;
    }

    // 事前準備：全組み合わせを列挙し、Gram・擬似逆行列を計算
    void prepare_subsets() {
        for(int sz : comb_size) {
            // 再帰的に組み合わせを作成
            vector<int> comb(sz);
            function<void(int, int)> dfs = [&](int start, int depth) {
                if(depth == sz) {
                    SubsetInfo info;
                    info.size = sz;
                    info.indices = comb;

                    // A_S^T * A_S = Gram (sz×sz)
                    info.Gram.assign(sz, vector<double>(sz, 0.0));
                    for(int i = 0; i < sz; i++) {
                        for(int j = 0; j < sz; j++) {
                            double dot = 0.0;
                            for(int d = 0; d < 3; d++) {
                                dot += paints[comb[i]][d] * paints[comb[j]][d];
                            }
                            info.Gram[i][j] = dot;
                        }
                    }

                    // Gram の逆行列 invG を計算
                    vector<vector<double>> invG;
                    invertMatrix(info.Gram, invG, sz);

                    // 擬似逆行列 = invG × A_S^T (sz×3)
                    info.pseudo.assign(sz, vector<double>(3, 0.0));
                    for(int i = 0; i < sz; i++) {
                        for(int d = 0; d < 3; d++) {
                            double sum = 0.0;
                            for(int j = 0; j < sz; j++) {
                                sum += invG[i][j] * paints[comb[j]][d];
                            }
                            info.pseudo[i][d] = sum;
                        }
                    }

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

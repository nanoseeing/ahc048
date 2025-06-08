
#pragma once

#include "common.hpp"
#include "game.hpp"
#include "utils.hpp"

// ====================================
// NNLSを解くためのクラス
// ====================================

class GreedyMixer {
  public:
    struct Result {
        double err;
        vector<int> indices;

        bool operator<(Result const& o) const {
            return err < o.err;
        }
    };

    Input& input;
    unordered_map<pair<int, int>, Result> results_greedy_cache;      // key:(h, comb_size), value: Result
    unordered_map<int, vector<Color>> mixed_colors_cache;            // key: comb_size, value: mixed colors
    unordered_map<int, vector<vector<int>>> mixed_colors_inds_cache; // key: comb_size, value: mixed color indices

    static constexpr int GREEDY_COLOR_MIN = 1; // greedyで混合する最小色数
    int GREEDY_COLOR_MAX;

    GreedyMixer(Input& input_, int GREEDY_COLOR_MAX_) : input(input_), GREEDY_COLOR_MAX(GREEDY_COLOR_MAX_) {
        construct_mixed_color();
        construct_greedy_policy();
    }

    int get_color_max() {
        return min(input.K, GREEDY_COLOR_MAX);
    }

    Result get_greedy_result(int h, int comb_size) {
        assert(0 <= h && h < input.H);
        assert(GREEDY_COLOR_MIN <= comb_size && comb_size <= get_color_max());
        pair<int, int> key = {h, comb_size};
        return results_greedy_cache[key];
    }

    vector<Color> get_mixed_colors(int comb_size) {
        assert(GREEDY_COLOR_MIN <= comb_size && comb_size <= get_color_max());
        return mixed_colors_cache[comb_size];
    }

    void construct_mixed_color() {
        int max_comb_size = min(input.K, GREEDY_COLOR_MAX);
        for(int comb_size = GREEDY_COLOR_MIN; comb_size <= max_comb_size; ++comb_size) {
            auto subsets = construct_subsets(comb_size, input.K);
            for(const auto& subset : subsets) {
                Color mixed_color = {0.0, 0.0, 0.0};
                for(int c = 0; c < 3; ++c) {
                    for(const int i : subset) {
                        mixed_color[c] += input.own[i][c];
                    }
                    mixed_color[c] /= (double)comb_size;
                }
                mixed_colors_cache[comb_size].push_back(mixed_color);
                mixed_colors_inds_cache[comb_size].push_back(subset);
            }
        }
    }

    void construct_greedy_policy() {
        int max_comb_size = min(input.K, GREEDY_COLOR_MAX);
        for(int comb_size = GREEDY_COLOR_MIN; comb_size <= max_comb_size; ++comb_size) {
            vector<double> best_costs(input.H, 1e9);
            vector<int> best_subset_inds(input.H, -1);

            for(int subi : range((int)mixed_colors_cache[comb_size].size())) {
                const auto& mixed_color = mixed_colors_cache[comb_size][subi];
                for(int h : range(input.H)) {
                    Color& target_color = input.target[h];
                    double err = eval_error(mixed_color, target_color);
                    if(err < best_costs[h]) {
                        best_costs[h] = err;
                        best_subset_inds[h] = subi;
                    }
                }
            }
            for(int h : range(input.H)) {
                Result r = {best_costs[h], mixed_colors_inds_cache[comb_size][best_subset_inds[h]]};
                results_greedy_cache[{h, comb_size}] = move(r);
            }
        }

        // 少ないターン数でエラーが小さければ採用する
        for(int h = 0; h < input.H; ++h) {
            for(int comb_size = GREEDY_COLOR_MIN + 1; comb_size <= max_comb_size; ++comb_size) {
                auto& pre_result = results_greedy_cache[{h, comb_size - 1}];
                auto& now_result = results_greedy_cache[{h, comb_size}];
                if(pre_result.err < now_result.err) {
                    results_greedy_cache[{h, comb_size}] = pre_result;
                }
            }
        }
    }
};

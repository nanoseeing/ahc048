#include "hpp/common.hpp"
#include "hpp/game.hpp"
#include "hpp/io.hpp"
#include "hpp/nnls_pdm.hpp"
#include "hpp/utils.hpp"

// ============================================================================
// 定義
// ============================================================================

const double MAX_TIME = 2800.0;

const int INIT_PARTITION_POS = 2; // パーティション初期値
const int SWITCH_EVAL_COST_TURN = 985;

const int TOP_N = 10000;
const int MAX_RESULT = 20;

const int BUFFER_TURN = 30; // 30ターンは余裕を持たせる

const double SWITH_POLICY_OBJ_TURN = 9.0;
const int MAX_APPLY_FRACTOR = 1;           // 分数適応回数の最大値
const vector<int> APPLY_FACTOR_LIST = {0}; // {0, 1}

// ============================================================================
// Main
// ============================================================================

struct GroupInfo {
    int k;
    int row_num;
    int start_x;
    std::vector<std::pair<int, int>> roots;
    int now_pos;
    int size;
};

class ManageGroupInfo {
  private:
    int n;
    int k;
    int original_k;
    int init_pos;
    std::vector<GroupInfo> infos;

    std::vector<std::pair<int, int>> create_root(int x, int row_num) {
        std::vector<std::pair<int, int>> roots;
        roots.emplace_back(n - 1, x);

        for(int r = 0; r < row_num; ++r) {
            if(r % 2 == 0) {
                for(int i = n - 2; i >= 0; --i) {
                    roots.emplace_back(i, x + r);
                }
            } else {
                for(int i = 0; i < n - 1; ++i) {
                    roots.emplace_back(i, x + r);
                }
            }
        }

        std::reverse(roots.begin(), roots.end());
        return roots;
    }

    std::vector<GroupInfo> construct_group_info() {
        std::vector<int> num_list(k, 1);
        for(int i = 0; i < n - k; ++i) {
            num_list[0] += 1;
            std::sort(num_list.begin(), num_list.end());
        }

        // acc_num_list: 累積和
        std::vector<int> acc_num_list(k, 0);
        for(int i = 0; i < k - 1; ++i) {
            acc_num_list[i + 1] = num_list[i] + acc_num_list[i];
        }

        std::vector<GroupInfo> result;
        result.reserve(k);
        for(int ki = 0; ki < k; ++ki) {
            int row_num = num_list[ki];
            int start_x = acc_num_list[ki];
            auto roots = create_root(start_x, row_num);

            GroupInfo info;
            info.k = ki;
            info.row_num = row_num;
            info.start_x = start_x;
            info.roots = std::move(roots);
            info.now_pos = init_pos;
            info.size = static_cast<int>(info.roots.size()) - 1;
            result.push_back(std::move(info));
        }

        return result;
    }

  public:
    ManageGroupInfo(int n_, int k_, int original_k_, int init_pos_ = 2) : n(n_), k(k_), original_k(original_k_), init_pos(init_pos_) {
        infos = construct_group_info();
    }

    int get_start_x(int k_index) const {
        return infos[k_index].start_x;
    }

    int get_now_pos(int k_index) const {
        return infos[k_index].now_pos;
    }

    int get_size(int k_index) const {
        return infos[k_index].size;
    }
    std::tuple<int, int, int, int> get_partition_pos(int k_index, int num) const {
        assert(0 < num && num <= infos[k_index].size);
        auto [y1, x1] = infos[k_index].roots[num - 1];
        auto [y2, x2] = infos[k_index].roots[num];
        return {y1, x1, y2, x2};
    }

    void change_now_pos(int k_index, int pos) {
        infos[k_index].now_pos = pos;
    }

    Action get_toggle_action(int k_index, int num) const {
        auto [y1, x1, y2, x2] = this->get_partition_pos(k_index, num);
        return Action::Toggle(y1, x1, y2, x2);
    }

    Action get_add_paint_action(int k_index) const {
        auto [y, x] = infos[k_index].roots[0];
        return Action::Add(y, x, k_index % this->original_k);
    }

    Action get_deliver_paint_action(int k_index) const {
        auto [y, x] = infos[k_index].roots[0];
        return Action::Deliver(y, x);
    }

    Paint get_paint(int k_index, const State &state) const {
        auto [y, x] = infos[k_index].roots[0];
        auto paint = state.get_paint(y, x);
        return paint;
    }

    Wall struct_init_wall(Input &input_data) {
        vector<vector<bool>> wall_h(input_data.N - 1, vector<bool>(input_data.N, false));
        vector<vector<bool>> wall_v(input_data.N, vector<bool>(input_data.N - 1, false));

        for(int x : range(input_data.N - 1)) {
            for(int y : range(input_data.N - 1)) {
                wall_v[y][x] = true;
            }
        }
        for(int x : range(input_data.N)) {
            wall_h[input_data.N - 2][x] = true;
        }

        // ルート間の仕切りを外す
        for(int k : range(input_data.K)) {
            const int root_size = (int)infos[k].roots.size();
            for(int i : range(1, root_size)) {
                auto [y1, x1] = infos[k].roots[i - 1];
                auto [y2, x2] = infos[k].roots[i];
                if(y1 == y2) {
                    wall_v[y1][min(x1, x2)] = false;
                } else {
                    wall_h[min(y1, y2)][x1] = false;
                }
            }
        }

        // 初期のパーティション位置を設定
        for(int k : range(input_data.K)) {
            auto [y1, x1, y2, x2] = this->get_partition_pos(k, this->init_pos);
            if(y1 == y2) {
                wall_v[y1][min(x1, x2)] = true;
            } else {
                wall_h[min(y1, y2)][x1] = true;
            }
        }

        // 混合する仕切りを開けておく
        for(int k : range(input_data.K)) {
            int s = this->get_size(k);
            auto [y1, x1, y2, x2] = this->get_partition_pos(k, s);
            assert(x1 == x2);
            wall_h[min(y1, y2)][x1] = false;
        }

        return Wall(wall_h, wall_v);
    }
};

void PolicyGreedy(Input &input, State &state, ManageGroupInfo &group_info) {
    auto target_color = input.target[state.deliver_cnt];

    double min_cost = 1e9;
    int min_k = -1;
    bool is_add = false;
    for(int k : range(state.input.K)) {
        Paint paint = group_info.get_paint(k, state);
        double now_cost = eval_error(input.own[k], target_color) * 1e4;
        bool now_is_add = (paint.vol < 1.0 - 1e-6) ? true : false; // 1g未満なら追加が必要
        if(now_is_add) now_cost += state.input.D;
        if(now_cost < min_cost) {
            min_cost = now_cost;
            min_k = k;
            is_add = now_is_add;
        }
    }

    if(is_add) {
        state.apply(group_info.get_add_paint_action(min_k));
    }

    state.apply(group_info.get_deliver_paint_action(min_k));
}

struct ImmediateInfo {
    int k;
    bool is_add;
    double rate;
    double vol;
    Fractors fractors;
};

struct DicisionAction {
    vector<Action> pre_actions;
    vector<Action> release_actions;
    vector<Action> post_actions;
    int act_cnt;
    int change_color_num;
};

class DicisionActionPerResult {
  private:
    struct RateItem {
        int k;
        double init_vol;
        // fractorの適用回数に応じて
        vector<vector<Fractors>> merge_fractors;
        vector<vector<double>> merge_rates;
        vector<vector<double>> merge_vols;
        vector<vector<bool>> merge_is_add;
    };

  public:
    vector<RateItem> rate_items;

    DicisionActionPerResult(State &state, Input &input, ManageGroupInfo &group_info) {
        this->construct(state, input, group_info);
    }

    void construct(State &state, Input &input, ManageGroupInfo &group_info) {
        this->rate_items.resize(input.K);

        for(int k : range(input.K)) {
            auto paint = group_info.get_paint(k, state);
            auto now_partition_pos = group_info.get_now_pos(k);
            auto max_rate = group_info.get_size(k);

            vector<Fractors> fractors;
            fractors.push_back({make_pair(1, 1)});   // 全開放
            fractors.push_back({make_pair(-1, -1)}); // 何もしない

            set<Fractor> fractor_set;
            for(int fractor_cnt : range(MAX_APPLY_FRACTOR)) {
                // 分数1回適応
                if(fractor_cnt == 0) {
                    for(int denominator : range(now_partition_pos, max_rate + 1)) {
                        for(int numerator : range(1, denominator)) {
                            Fractor fractor = make_pair(numerator, denominator);
                            Fractor reduced_fractor = reduce_fraction(fractor);
                            if(fractor_set.contains(reduced_fractor)) continue;
                            fractor_set.insert(reduced_fractor);
                            fractors.push_back({fractor});
                        }
                    }
                } else if(fractor_cnt == 1) {
                    // 分数2回適応
                    set<Fractor> fractor_set;
                    for(int d1 : range(now_partition_pos + 1, max_rate + 1)) {
                        for(int n1 : range(1, d1)) {
                            Fractor f1 = make_pair(n1, d1);
                            // 次の分母の最小値 = 下側のブロック数 = 前の分子
                            // 次の分母の最大値 += 最大ブロック数 - 前の分母
                            for(int d2 : range(max(2, n1), n1 + (max_rate - d1) + 1)) {
                                for(int n2 : range(1, d2)) {
                                    Fractor f2 = make_pair(n2, d2);
                                    Fractor fractor = mul_fracs({f1, f2});
                                    if(fractor_set.contains(fractor)) continue;
                                    fractor_set.insert(fractor);
                                    fractors.push_back({f1, f2});
                                }
                            }
                        }
                    }
                }

                this->rate_items[k].k = k;
                this->rate_items[k].init_vol = paint.vol;

                // fractorからrateなどを計算
                vector<double> rates;
                vector<double> vols;
                vector<double> vols_add;
                for(auto &fractor_vec : fractors) {
                    double rate = 1.0;
                    for(auto &fractor : fractor_vec) {
                        if(fractor.first == -1) {
                            rate = 0.0;
                        } else {
                            rate *= (double)(fractor.first) / (double)(fractor.second);
                        }
                    }
                    rates.emplace_back(rate);
                    vols.emplace_back(paint.vol * rate);
                    vols_add.emplace_back((paint.vol + 1.0) * rate);
                }

                // ソートしておく
                auto inds = make_sorted_indices(rates);
                reorder_vector(fractors, inds);
                reorder_vector(rates, inds);
                reorder_vector(vols, inds);
                reorder_vector(vols_add, inds);

                auto false_vec = vector<bool>(rates.size(), false);
                auto true_vec = vector<bool>(rates.size(), true);
                if(paint.vol < 1.0) {
                    // 1.0g追加可能ならマージ
                    auto merge_fractors = fractors;
                    auto merge_rates = rates;
                    auto merge_vols = vols;
                    auto merge_is_add = false_vec;
                    merge_fractors.insert(merge_fractors.begin(), ALL(fractors));
                    merge_rates.insert(merge_rates.begin(), ALL(rates));
                    merge_vols.insert(merge_vols.begin(), ALL(vols_add));
                    merge_is_add.insert(merge_is_add.begin(), ALL(true_vec));

                    auto merge_inds = make_sorted_indices(merge_vols);
                    reorder_vector(merge_fractors, merge_inds);
                    reorder_vector(merge_rates, merge_inds);
                    reorder_vector(merge_vols, merge_inds);
                    reorder_vector(merge_is_add, merge_inds);

                    this->rate_items[k].merge_fractors.emplace_back(merge_fractors);
                    this->rate_items[k].merge_rates.emplace_back(merge_rates);
                    this->rate_items[k].merge_vols.emplace_back(merge_vols);
                    this->rate_items[k].merge_is_add.emplace_back(merge_is_add);
                } else {
                    this->rate_items[k].merge_fractors.emplace_back(fractors);
                    this->rate_items[k].merge_rates.emplace_back(rates);
                    this->rate_items[k].merge_vols.emplace_back(vols);
                    this->rate_items[k].merge_is_add.emplace_back(false_vec);
                }
            }
        }
    }

    double eval_cost(Input &input, State &state, vector<ImmediateInfo> &immeediate_info, ManageGroupInfo &group_info) {
        auto &now_target = state.input.target[state.deliver_cnt];

        double sum_vol = 0.0;
        int add_cnt = 0;
        vector<double> vols;
        vector<Color> colors;
        for(auto info : immeediate_info) {
            vols.emplace_back(info.vol);
            colors.emplace_back(input.own[info.k]);
            sum_vol += info.vol;
            if(info.is_add) add_cnt++;
        }

        Color mixed_color = mix(vols, colors);
        double err_cost = eval_error(mixed_color, now_target) * 1e4;
        double discard_cost = max(0.0, sum_vol - 1.0) * (double)(input.D);

        if(state.deliver_cnt <= SWITCH_EVAL_COST_TURN) {
            // 通常は、廃棄=追加コストとみなす
            return err_cost + discard_cost;
        } else {
            // 最後の方は、追加コストを見る

            // double total_board_vol = 0.0;
            // for(int k : range(input.K)) {
            //     auto paint = group_info.get_paint(k, state);
            //     total_board_vol += paint.vol;
            // }
            // double add_cost = (total_board_vol + add_cnt) * (double)(input.D);
            double add_cost = max(0, state.add_cnt + add_cnt - (state.deliver_cnt + 1)) * (double)(input.D);

            return err_cost + add_cost;
        }
    }

    int search_target_weight_idx(int k, double weight, int target_mul_cnt) {
        auto &rate_item = this->rate_items[k];
        int rate_item_size = (int)rate_item.merge_vols[target_mul_cnt].size();
        int it_ind = -1;
        auto it = upper_bound(ALL(rate_item.merge_vols[target_mul_cnt]), weight);
        it_ind = distance(rate_item.merge_vols[target_mul_cnt].begin(), it);
        if(it_ind >= rate_item_size) {
            it_ind = rate_item_size - 1;
        }
        return it_ind;
    }

    tuple<vector<ImmediateInfo>, double> eval_one_result(ColorMixer::Result &result, State &state, Input &input, int target_mul_cnt,
                                                         ManageGroupInfo &manage_group_info) {
        double best_cost = 1e9;
        vector<ImmediateInfo> best_info;

        int comb_size = result.indices.size();

        int first_k = result.indices[0];
        double first_target_weight = result.weights[0];
        int first_arr_size = this->rate_items[first_k].merge_vols[target_mul_cnt].size();
        int first_it_ind = search_target_weight_idx(first_k, first_target_weight, target_mul_cnt);
        if(first_it_ind != 0) {
            first_it_ind--;
        }

        // weightsの係数を守ること優先
        vector<vector<ImmediateInfo>> infos;
        for(int comb_ind : range(comb_size)) {
            auto k = result.indices[comb_ind];
            auto target_weight = result.weights[comb_ind];
            auto &rate_item = rate_items[k];
            double targe_vol = target_weight;
            auto it_ind = search_target_weight_idx(k, targe_vol, target_mul_cnt);
            vector<ImmediateInfo> immediate_infos;

            int left, right;
            left = -1, right = 2; // 初期値
            for(int j : range(left, right)) {
                if(it_ind + j < 0 || it_ind + j >= (int)rate_item.merge_vols[target_mul_cnt].size()) continue;
                int new_ind = it_ind + j;
                auto &vol = rate_item.merge_vols[target_mul_cnt][new_ind];
                auto &rate = rate_item.merge_rates[target_mul_cnt][new_ind];
                auto &fractors = rate_item.merge_fractors[target_mul_cnt][new_ind];
                bool is_add = rate_item.merge_is_add[target_mul_cnt][new_ind];
                ImmediateInfo info = {.k = k, .is_add = is_add, .rate = rate, .vol = vol, .fractors = fractors};
                // cpp_dump(boost::format("k: %d, vol: %.6f, rate: %.6f, is_add: %d, target_weight: %.6f") % k % vol % rate % is_add % target_weight);
                immediate_infos.emplace_back(info);
            }
            infos.emplace_back(immediate_infos);
        }

        cartesian_product(infos, [&](vector<ImmediateInfo> &comb) {
            double sum_vol = 0.0;
            for(const auto &info : comb) {
                sum_vol += info.vol;
            }
            if(sum_vol > 1.0 - 1e-6) {
                double cost = eval_cost(input, state, comb, manage_group_info);
                if(cost < best_cost) {
                    best_cost = cost;
                    best_info = comb;
                }
            }
        });

        return {best_info, best_cost};
    }
};

DicisionAction construct_from_immediateinfo(vector<ImmediateInfo> &best_info, ManageGroupInfo &manage_group_info, State &state,
                                            ColorMixer::Result &mix_result) {
    DicisionAction action_result;
    action_result.change_color_num = (int)best_info.size();

    // for(int i : mix_result.indices) {
    //     cerr << boost::format("%.6f - %.6f") % mix_result.weights[i] % best_info[i].vol << " | ";
    // }
    // cerr << endl;

    for(auto &info : best_info) {
        int now_partition_pos = manage_group_info.get_now_pos(info.k);
        int frac_size = info.fractors.size();

        // 追加は最初にやっておく
        if(info.is_add) {
            action_result.pre_actions.emplace_back(manage_group_info.get_add_paint_action(info.k));
        }

        auto &first_fractor = info.fractors[0];
        if(first_fractor.first == -1 && first_fractor.second == -1) {
            continue;
        } else if(first_fractor.first == 1 && first_fractor.second == 1) {
            // 全開放
            assert(frac_size == 1);
            action_result.release_actions.emplace_back(manage_group_info.get_toggle_action(info.k, now_partition_pos));
            action_result.post_actions.emplace_back(manage_group_info.get_toggle_action(info.k, INIT_PARTITION_POS));
            manage_group_info.change_now_pos(info.k, INIT_PARTITION_POS);
        } else {
            int upper_partition = 0;
            int lower_partition = now_partition_pos;

            // 分割n回適応
            for(int fi : range(frac_size)) {
                auto &fractor = info.fractors[fi];

                // 上の仕切りから、分母だけ進んだのがstopしたいしきり位置
                int stop_par_pos = upper_partition + fractor.second;
                // stopする仕切りから、分子だけ進んだのが、releaseする仕切り位置
                int release_par_pos = stop_par_pos - fractor.first;

                if(stop_par_pos != lower_partition) {
                    // 現在の仕切りを動かす必要があるなら、仕切りを拡張する
                    action_result.pre_actions.emplace_back(manage_group_info.get_toggle_action(info.k, stop_par_pos));
                    action_result.pre_actions.emplace_back(manage_group_info.get_toggle_action(info.k, lower_partition));
                }
                // 分子の位置で止める
                action_result.pre_actions.emplace_back(manage_group_info.get_toggle_action(info.k, release_par_pos));

                if(fi == frac_size - 1) {
                    // 分母の位置で解放する
                    action_result.release_actions.emplace_back(manage_group_info.get_toggle_action(info.k, stop_par_pos));
                    // 最後の仕切り位置は、release地点になる
                    manage_group_info.change_now_pos(info.k, release_par_pos);
                } else {
                    // 仮止めした仕切りは解放しておく必要がある
                    action_result.post_actions.emplace_back(manage_group_info.get_toggle_action(info.k, release_par_pos));
                }

                upper_partition = release_par_pos;
                lower_partition = stop_par_pos;
            }
            if(manage_group_info.get_now_pos(info.k) == 1) {
                // 仕切りが小さくなったら復活しておく
                int now_pos = manage_group_info.get_now_pos(info.k);
                action_result.post_actions.emplace_back(manage_group_info.get_toggle_action(info.k, INIT_PARTITION_POS));
                action_result.post_actions.emplace_back(manage_group_info.get_toggle_action(info.k, now_pos));
                manage_group_info.change_now_pos(info.k, INIT_PARTITION_POS);
            }
        }
    }
    int act_cnt = action_result.pre_actions.size() + action_result.release_actions.size() + action_result.post_actions.size();
    action_result.act_cnt = act_cnt;

    return action_result;
}

DicisionAction dicision_action(Input &input, State &state, ColorMixer &mixer, double obj_turn, ManageGroupInfo &group_info) {
    Color target = input.target[state.deliver_cnt];
    vector<int> indices;
    for(int k : range(input.K)) {
        indices.push_back(k);
    }
    auto single_result = mixer.solve_nnls_for_indices(indices, target);
    // cpp_dump(single_result.squared_error * 1e4);

    vector<ColorMixer::Result> results = {single_result};
    // for(const auto &result : results) {
    //     int comb_size = result.indices.size();
    //     double pred_turn = comb_size * 4.0 + 1.0; // 1色あたり4ターン + 2.0ターンのバッファ
    //     if(pred_turn <= obj_turn) {
    //         results.emplace_back(result);
    //     }
    //     if((int)results.size() >= MAX_RESULT) {
    //         break;
    //     }
    // }

    double best_cost = 1e9;
    vector<ImmediateInfo> best_info;

    DicisionActionPerResult per_result = DicisionActionPerResult(state, input, group_info);
    for(auto &result : results) {
        // ! DEBUG
        for(int target_mul_cnt : APPLY_FACTOR_LIST) {
            double pred_turn = result.indices.size() * (target_mul_cnt + 1) * 4.0 + 1.0;
            if(pred_turn > obj_turn) {
                continue; // 目標ターン数を超える場合はスキップ
            }
            auto [now_info, now_cost] = per_result.eval_one_result(result, state, input, target_mul_cnt, group_info);
            if(now_cost < best_cost) {
                best_cost = now_cost;
                best_info = now_info;
            }
        }
    }

    assert((int)best_info.size() != 0);

    auto action_result = construct_from_immediateinfo(best_info, group_info, state, single_result);
    return action_result;
}

void discard_mix_well(State &state, Input &input) {
    while(state.get_paint(input.N - 1, 0).vol > 1e-6) {
        state.apply(Action::Discard(input.N - 1, 0));
    }
}

void print_info(State &state) {
    auto [deliver_cost, err_cost, total_cost] = state.get_score();
    cerr << boost::format("H: %d, Turn: %d, Add %d, Score: %d (add: %d, err: %d)") % state.deliver_cnt % state.turn % state.add_cnt % total_cost %
                deliver_cost % err_cost
         << endl;
}

void solve() {
    TimeKeeper time_keeper(MAX_TIME);
    Input input = parse_input();

    ManageGroupInfo manage_group_info(input.N, input.K, input.K, INIT_PARTITION_POS);
    auto init_wall = manage_group_info.struct_init_wall(input);
    State state(init_wall, input);
    ColorMixer mixer(input.own);

    // Main Loop
    int policy_greedy_cnt = 0;
    double policy_err_sum = 0.0;
    map<int, int> act_cnt;
    map<int, int> color_cnt;

    try {
        for(int h : range(input.H)) {
            // if(h % 10 == 0) print_info(state);
            print_info(state);

            int remain_turn = input.T - state.turn - BUFFER_TURN;
            double obj_turn = (double)remain_turn / (double)(input.H - state.deliver_cnt);

            if(obj_turn >= SWITH_POLICY_OBJ_TURN) {
                auto action_result = dicision_action(input, state, mixer, obj_turn, manage_group_info);

                // tmp
                act_cnt[action_result.change_color_num] += action_result.act_cnt;
                color_cnt[action_result.change_color_num]++;
                //

                for(const auto &act : action_result.pre_actions) {
                    state.apply(act);
                }
                for(const auto &act : action_result.release_actions) {
                    state.apply(act);
                }
                state.apply(Action::Deliver(input.N - 1, 0));
                discard_mix_well(state, input);
                for(const auto &act : action_result.post_actions) {
                    state.apply(act);
                }

            } else {
                auto pre_err = state.error;
                PolicyGreedy(input, state, manage_group_info);
                auto post_err = state.error;
                policy_err_sum += post_err - pre_err;
                policy_greedy_cnt++;
            }
        }
        print_info(state);
    } catch(const exception &e) {
        Output output = Output{init_wall, state.actions};
        print_output(output);
        cerr << "Exception: " << e.what() << endl;
        exit(1);
    }

    // 情報
    if(policy_greedy_cnt > 0) {
        cerr << boost::format("PolicyGreedy %d times. Error %d") % policy_greedy_cnt % (policy_err_sum * 1e4) << endl;
    }
    for(const auto &p : act_cnt) {
        int color_num = p.first;
        int call = color_cnt[color_num];
        int total = p.second;
        double avg = (double)total / (double)call;
        cerr << boost::format("color num: %d, call: %d, total: %d, avg: %f") % color_num % call % total % avg << endl;
    }
    cerr << boost::format("score: %d, elapsed: %f, turn: %d/%d]") % get<2>(state.get_score()) % time_keeper.getElapsedTime() % state.turn % input.T << endl;
    Output output = Output{init_wall, state.actions};
    print_output(output);
}

int main() {
    solve();
    return 0;
}
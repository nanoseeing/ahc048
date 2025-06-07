#include "hpp/comb.hpp"
#include "hpp/common.hpp"
#include "hpp/game.hpp"
#include "hpp/io.hpp"
#include "hpp/utils.hpp"

// ============================================================================
// 定義
// ============================================================================

const double MAX_TIME = 2800.0;

const int INIT_PARTITION_POS = 1; // パーティション初期値
const int MAX_RATE = 19;          // パーティション最大値
const int SWITCH_EVAL_COST_TURN = 990;

const int TOP_N = 10000;
const int MAX_RESULT = 20;

const int BUFFER_TURN = 30; // 30ターンは余裕を持たせる

const double SWICH_POLICY_OBJ_TURN = 9.0;
const int MAX_APPLY_FRACTOR = 1; // 分数適応回数の最大値
const int TARGET_MUL_CNT = 0;    // 何回分数を適応するか（tmp)

// ============================================================================
// Main
// ============================================================================

struct PrimitiveColorGroupInfo {
    vector<int> partition_positions;
};

tuple<Wall, PrimitiveColorGroupInfo> struct_init_wall(Input &input_data) {
    vector<vector<bool>> wall_h(input_data.N - 1, vector<bool>(input_data.N, false));
    vector<vector<bool>> wall_v(input_data.N, vector<bool>(input_data.N - 1, false));

    for(int x : range(input_data.N - 1)) {
        for(int y : range(input_data.N - 1)) {
            wall_v[y][x] = true;
        }
    }

    for(int x : range(input_data.N)) {
        wall_h[INIT_PARTITION_POS][x] = true;
    }

    vector<int> partition_positions(input_data.K, INIT_PARTITION_POS);
    PrimitiveColorGroupInfo primitive_color_group_info;
    primitive_color_group_info.partition_positions = partition_positions;

    return {Wall(wall_h, wall_v), primitive_color_group_info};
}

void PolicyGreedy(Input &input, State &state) {
    auto target_color = input.target[state.deliver_cnt];

    double min_cost = 1e9;
    int min_k = -1;
    bool is_add = false;
    for(int k : range(state.input.K)) {
        Paint paint = state.get_paint(0, k);
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
        state.apply(Action::Add(0, min_k, min_k));
    }
    state.apply(Action::Deliver(0, min_k));
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

    DicisionActionPerResult(State &state, Input &input, PrimitiveColorGroupInfo &primitive_group_info) {
        this->construct(state, input, primitive_group_info);
    }

    void construct(State &state, Input &input, PrimitiveColorGroupInfo &primitive_group_info) {
        this->rate_items.resize(input.K);

        for(int k : range(input.K)) {
            auto paint = state.get_paint(0, k);
            auto now_partition_pos = primitive_group_info.partition_positions[k];
            for(int fractor_cnt : range(MAX_APPLY_FRACTOR)) {
                // 分数1回適応
                vector<Fractors> fractors;
                if(fractor_cnt == 0) {
                    fractors.push_back({make_pair(1, 1)}); // 全開放を許す
                    set<Fractor> fractor_set;
                    for(int denominator : range(now_partition_pos + 1, MAX_RATE + 1)) {
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
                    for(int d1 : range(now_partition_pos + 2, MAX_RATE + 1)) {
                        for(int n1 : range(1, d1)) {
                            Fractor f1 = make_pair(n1, d1);
                            // 次の分母の最小値 = 下側のブロック数 = 前の分子
                            // 次の分母の最大値 += 最大ブロック数 - 前の分母
                            for(int d2 : range(max(2, n1), n1 + (MAX_RATE - d1) + 1)) {
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
                        rate *= (double)(fractor.first) / (double)(fractor.second);
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

    double eval_cost(Input &input, State &state, vector<ImmediateInfo> &immeediate_info) {
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
            double total_board_vol = 0.0;
            for(int k : range(input.K)) {
                auto paint = state.get_paint(0, k);
                total_board_vol += paint.vol;
            }

            double add_cost = max(0, state.add_cnt + add_cnt - (state.deliver_cnt + 1)) * (double)(input.D);
            // double add_cost = (total_board_vol + add_cnt) * (double)(input.D);

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

    tuple<vector<ImmediateInfo>, double> eval_one_result(ColorMixer::Result &result, State &state, Input &input) {
        double best_cost = 1e9;
        vector<ImmediateInfo> best_info;

        int comb_size = result.indices.size();

        int first_k = result.indices[0];
        double first_target_weight = result.weights[0];
        int first_arr_size = this->rate_items[first_k].merge_vols[TARGET_MUL_CNT].size();
        int first_it_ind = search_target_weight_idx(first_k, first_target_weight, TARGET_MUL_CNT);
        if(first_it_ind != 0) {
            first_it_ind--;
        }

        // 誤差を減らす方優先
        for(int ind : range(first_it_ind, min(first_arr_size, first_it_ind + 10))) {
            vector<vector<ImmediateInfo>> infos;

            auto &first_vol = this->rate_items[first_k].merge_vols[TARGET_MUL_CNT][ind];
            auto &first_rate = this->rate_items[first_k].merge_rates[TARGET_MUL_CNT][ind];
            auto &first_fractors = this->rate_items[first_k].merge_fractors[TARGET_MUL_CNT][ind];
            bool first_is_add = this->rate_items[first_k].merge_is_add[TARGET_MUL_CNT][ind];
            ImmediateInfo first_info = {.k = first_k, .is_add = first_is_add, .rate = first_rate, .vol = first_vol, .fractors = first_fractors};
            infos.push_back({first_info});

            double target_mul = first_vol / first_target_weight;
            for(int comb_ind : range(1, comb_size)) {
                auto k = result.indices[comb_ind];
                auto target_weight = result.weights[comb_ind];
                auto &rate_item = rate_items[k];
                double targe_vol = target_weight * target_mul;
                auto it_ind = search_target_weight_idx(k, targe_vol, TARGET_MUL_CNT);
                vector<ImmediateInfo> immediate_infos;
                for(int j : range(-1, 1)) {
                    if(it_ind + j < 0 || it_ind + j >= (int)rate_item.merge_vols[TARGET_MUL_CNT].size()) continue;
                    int new_ind = it_ind + j;
                    auto &vol = rate_item.merge_vols[TARGET_MUL_CNT][new_ind];
                    auto &rate = rate_item.merge_rates[TARGET_MUL_CNT][new_ind];
                    auto &fractors = rate_item.merge_fractors[TARGET_MUL_CNT][new_ind];
                    bool is_add = rate_item.merge_is_add[TARGET_MUL_CNT][new_ind];
                    ImmediateInfo info = {.k = k, .is_add = is_add, .rate = rate, .vol = vol, .fractors = fractors};
                    immediate_infos.emplace_back(info);
                }
                infos.emplace_back(immediate_infos);
            }

            cartesian_product(infos, [&](vector<ImmediateInfo> &comb) {
                double sum_vol = 0.0;
                for(const auto &info : comb) {
                    sum_vol += info.vol;
                }
                if(sum_vol >= 1.0 - 9.99e-7) { // 1e-6だと誤差が怖い
                    double cost = eval_cost(input, state, comb);
                    if(cost < best_cost) {
                        best_cost = cost;
                        best_info = comb;
                    }
                }
            });
        }

        // weightsの係数を守ること優先
        vector<vector<ImmediateInfo>> infos;
        for(int comb_ind : range(comb_size)) {
            auto k = result.indices[comb_ind];
            auto target_weight = result.weights[comb_ind];
            auto &rate_item = rate_items[k];
            double targe_vol = target_weight;
            auto it_ind = search_target_weight_idx(k, targe_vol, TARGET_MUL_CNT);
            vector<ImmediateInfo> immediate_infos;

            int left, right;
            if(comb_size == 4) {
                left = -2, right = 2;
            } else if(comb_size == 3) {
                left = -4, right = 4;
            } else {
                left = -10, right = 10;
            }

            for(int j : range(left, right)) {
                if(it_ind + j < 0 || it_ind + j >= (int)rate_item.merge_vols[TARGET_MUL_CNT].size()) continue;
                int new_ind = it_ind + j;
                auto &vol = rate_item.merge_vols[TARGET_MUL_CNT][new_ind];
                auto &rate = rate_item.merge_rates[TARGET_MUL_CNT][new_ind];
                auto &fractors = rate_item.merge_fractors[TARGET_MUL_CNT][new_ind];
                bool is_add = rate_item.merge_is_add[TARGET_MUL_CNT][new_ind];
                ImmediateInfo info = {.k = k, .is_add = is_add, .rate = rate, .vol = vol, .fractors = fractors};
                immediate_infos.emplace_back(info);
            }
            infos.emplace_back(immediate_infos);
        }

        cartesian_product(infos, [&](vector<ImmediateInfo> &comb) {
            double sum_vol = 0.0;
            for(const auto &info : comb) {
                sum_vol += info.vol;
            }
            if(sum_vol >= 1.0 - 9.99e-7) { // 1e-6だと誤差が怖い
                double cost = eval_cost(input, state, comb);
                if(cost < best_cost) {
                    best_cost = cost;
                    best_info = comb;
                }
            }
        });

        return {best_info, best_cost};
    }
};

tuple<DicisionAction, PrimitiveColorGroupInfo> construct_from_immediateinfo(vector<ImmediateInfo> &best_info, PrimitiveColorGroupInfo &primitive_group_info) {
    DicisionAction action_result;
    PrimitiveColorGroupInfo new_group_info = primitive_group_info;
    action_result.change_color_num = (int)best_info.size();

    for(auto &info : best_info) {
        int now_partition_pos = primitive_group_info.partition_positions[info.k];
        int frac_size = info.fractors.size();

        // 追加は最初にやっておく
        if(info.is_add) {
            action_result.pre_actions.emplace_back(Action::Add(0, info.k, info.k));
        }

        auto &first_fractor = info.fractors[0];
        if(first_fractor.first == 1 && first_fractor.second == 1) {
            // 全開放
            assert(frac_size == 1);
            action_result.release_actions.emplace_back(Action::Toggle(now_partition_pos, info.k, now_partition_pos + 1, info.k));
            action_result.post_actions.emplace_back(Action::Toggle(INIT_PARTITION_POS, info.k, INIT_PARTITION_POS + 1, info.k));
            new_group_info.partition_positions[info.k] = INIT_PARTITION_POS;
        } else {
            int upper_partition = 0;
            int lower_partition = now_partition_pos + 1;

            // 分割n回適応
            for(int fi : range(frac_size)) {
                auto &fractor = info.fractors[fi];

                // 上の仕切りから、分母だけ進んだのがstopしたいしきり位置
                int stop_par_pos = upper_partition + fractor.second;
                // stopする仕切りから、分子だけ進んだのが、releaseする仕切り位置
                int release_par_pos = stop_par_pos - fractor.first;

                if(stop_par_pos != lower_partition) {
                    // 現在の仕切りを動かす必要があるなら、仕切りを拡張する
                    action_result.pre_actions.emplace_back(Action::Toggle(stop_par_pos - 1, info.k, stop_par_pos, info.k));
                    action_result.pre_actions.emplace_back(Action::Toggle(lower_partition - 1, info.k, lower_partition, info.k));
                }
                // 分子の位置で止める
                action_result.pre_actions.emplace_back(Action::Toggle(release_par_pos - 1, info.k, release_par_pos, info.k));

                if(fi == frac_size - 1) {
                    // 分母の位置で解放する
                    action_result.release_actions.emplace_back(Action::Toggle(stop_par_pos - 1, info.k, stop_par_pos, info.k));
                    // 最後の仕切り位置は、release地点になる
                    new_group_info.partition_positions[info.k] = release_par_pos - 1;
                } else {
                    // 仮止めした仕切りは解放しておく必要がある
                    action_result.post_actions.emplace_back(Action::Toggle(release_par_pos - 1, info.k, release_par_pos, info.k));
                }

                upper_partition = release_par_pos;
                lower_partition = stop_par_pos;
            }
            if(new_group_info.partition_positions[info.k] == 0) {
                // 仕切りが小さくなったら復活しておく
                int now_pos = new_group_info.partition_positions[info.k];
                action_result.post_actions.emplace_back(Action::Toggle(INIT_PARTITION_POS, info.k, INIT_PARTITION_POS + 1, info.k));
                action_result.post_actions.emplace_back(Action::Toggle(now_pos, info.k, now_pos + 1, info.k));
                new_group_info.partition_positions[info.k] = INIT_PARTITION_POS;
            }
        }
    }
    int act_cnt = action_result.pre_actions.size() + action_result.release_actions.size() + action_result.post_actions.size();
    action_result.act_cnt = act_cnt;

    return {action_result, new_group_info};
}

tuple<DicisionAction, PrimitiveColorGroupInfo> dicision_action(Input &input, State &state, ColorMixer &mixer, double obj_turn,
                                                               PrimitiveColorGroupInfo &primitive_group_info) {
    Color target = input.target[state.deliver_cnt];
    auto all_results = mixer.find_topN(target, TOP_N);

    vector<ColorMixer::Result> results;
    for(const auto &result : all_results) {
        int comb_size = result.indices.size();
        double pred_turn = comb_size * 4.0 + 1.0; // 1色あたり4ターン + 2.0ターンのバッファ
        if(pred_turn <= obj_turn) {
            results.emplace_back(result);
        }
        if((int)results.size() >= MAX_RESULT) {
            break;
        }
    }

    double best_cost = 1e9;
    vector<ImmediateInfo> best_info;

    DicisionActionPerResult per_result = DicisionActionPerResult(state, input, primitive_group_info);
    for(auto &result : results) {
        auto [now_info, now_cost] = per_result.eval_one_result(result, state, input);
        if(now_cost < best_cost) {
            best_cost = now_cost;
            best_info = now_info;
        }
    }

    assert((int)best_info.size() != 0);

    auto [action_result, new_group_info] = construct_from_immediateinfo(best_info, primitive_group_info);
    return {action_result, new_group_info};
}

void init_state_add_1gram(State &state, Input &input) {
    for(int x : range(input.K)) {
        state.apply(Action::Add(0, x, x));
    }
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
    auto [init_wall, primitive_group_info] = struct_init_wall(input);

    State state(init_wall, input);
    init_state_add_1gram(state, input);

    ColorMixer mixer(input.own);

    // Main Loop
    int policy_greedy_cnt = 0;
    double policy_err_sum = 0.0;
    map<int, int> act_cnt;
    map<int, int> color_cnt;

    try {
        for(int h : range(input.H)) {
            if(h % 10 == 0) print_info(state);

            int remain_turn = input.T - state.turn - BUFFER_TURN;
            double obj_turn = (double)remain_turn / (double)(input.H - state.deliver_cnt);

            if(obj_turn >= SWICH_POLICY_OBJ_TURN) {
                auto [action_result, new_group_info] = dicision_action(input, state, mixer, obj_turn, primitive_group_info);

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
                primitive_group_info = new_group_info;

            } else {
                auto pre_err = state.error;
                PolicyGreedy(input, state);
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
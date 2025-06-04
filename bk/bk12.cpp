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
const int MAX_RESULT = 18;

const int BUFFER_TURN = 30; // 30ターンは余裕を持たせる

const double SWITH_POLICY_OBJ_TURN = 10.0;
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
        vector<vector<Fractors>> fractors;
        vector<vector<double>> rates;
        vector<vector<double>> vols;
        vector<vector<double>> post_add_vols;
    };

  public:
    vector<RateItem> rate_items;

    DicisionActionPerResult(State &state, Input &input, PrimitiveColorGroupInfo &primitive_group_info) {
        this->construct(state, input, primitive_group_info);
    }

    void construct(State &state, Input &input, PrimitiveColorGroupInfo &primitive_group_info) {
        this->rate_items.resize(input.K);

        for(int k : range(input.K)) {
            Paint paint = state.get_paint(0, k);
            int now_partition_pos = primitive_group_info.partition_positions[k];
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

                // fractorからrateなどを計算
                vector<double> rates;
                vector<double> vols;
                vector<double> post_add_vols;
                for(auto &fractor_vec : fractors) {
                    double rate = 1.0;
                    for(auto &fractor : fractor_vec) {
                        rate *= (double)(fractor.first) / (double)(fractor.second);
                    }
                    rates.push_back(rate);
                    vols.push_back(paint.vol * rate);
                    post_add_vols.push_back((paint.vol + 1.0) * rate);
                }

                // ソートしておく
                vector<int> inds;
                for(int i : range(rates.size())) {
                    inds.push_back(i);
                }
                sort(ALL(inds), [&](int a, int b) { return rates[a] < rates[b]; });
                vector<Fractors> sorted_fractors(fractors.size());
                vector<double> sorted_rates(rates.size());
                vector<double> sorted_vols(vols.size());
                vector<double> sorted_post_add_vols(post_add_vols.size());
                for(int i : range(rates.size())) {
                    sorted_fractors[i] = fractors[inds[i]];
                    sorted_rates[i] = rates[inds[i]];
                    sorted_vols[i] = vols[inds[i]];
                    sorted_post_add_vols[i] = post_add_vols[inds[i]];
                }

                // 追加
                this->rate_items[k].fractors.push_back(sorted_fractors);
                this->rate_items[k].rates.push_back(sorted_rates);
                this->rate_items[k].vols.push_back(sorted_vols);
                this->rate_items[k].post_add_vols.push_back(sorted_post_add_vols);
                this->rate_items[k].k = k;
                this->rate_items[k].init_vol = paint.vol;
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
            vols.push_back(info.vol);
            colors.push_back(input.own[info.k]);
            sum_vol += info.vol;
            if(info.is_add) add_cnt++;
        }

        Color mixed_color = mix(vols, colors);
        double err_cost = eval_error(mixed_color, now_target) * 1e4;
        double discard_cost = max(0.0, sum_vol - 1.0) * (double)(input.D);
        double add_cost = max(0, state.add_cnt + add_cnt - (state.deliver_cnt + 1)) * (double)(input.D);

        if(state.deliver_cnt <= SWITCH_EVAL_COST_TURN) {
            // 通常は、廃棄=追加コストとみなす
            return err_cost + discard_cost;
        } else {
            // 最後の方は、純粋な追加コストだけみる
            return err_cost + add_cost;
        }
    }

    tuple<vector<ImmediateInfo>, double> eval_one_result(ColorMixer::Result &result, State &state, Input &input) {
        double best_cost = 1e9;
        vector<ImmediateInfo> best_info;

        int comb_size = result.indices.size();

        // weightsに近いvolumeを持つペイントのrateを求める(最後のペイントは無視)
        vector<vector<ImmediateInfo>> now_infos(comb_size - 1);
        for(int i : range(comb_size - 1)) {
            int k = result.indices[i];
            double weight = result.weights[i];
            auto &rate_item = rate_items[k];
            int it_ind = -1;
            int rate_item_size = (int)rate_item.rates[TARGET_MUL_CNT].size();
            bool is_add = weight > rate_item.init_vol;
            if(!is_add) {
                // addしない場合
                auto it = upper_bound(ALL(rate_item.vols[TARGET_MUL_CNT]), weight);
                it_ind = distance(rate_item.vols[TARGET_MUL_CNT].begin(), it);
            } else {
                // addする場合
                auto it = upper_bound(ALL(rate_item.post_add_vols[TARGET_MUL_CNT]), weight);
                it_ind = distance(rate_item.post_add_vols[TARGET_MUL_CNT].begin(), it);
            }
            if(it_ind >= rate_item_size) {
                it_ind = rate_item_size - 1;
            }

            // TMP
            int left, right;
            if(comb_size == 4) {
                left = -2, right = 2;
            } else if(comb_size == 3) {
                left = -4, right = 4;
            } else {
                left = -10, right = 10;
            }

            for(int j : range(left, right)) {
                if(it_ind + j < 0 || it_ind + j >= rate_item_size) continue;
                int new_ind = it_ind + j;
                ImmediateInfo info;
                info.k = k;
                info.is_add = is_add;
                info.vol = is_add ? rate_item.post_add_vols[TARGET_MUL_CNT][new_ind] : rate_item.vols[TARGET_MUL_CNT][new_ind];
                info.fractors = rate_item.fractors[TARGET_MUL_CNT][new_ind];
                info.rate = rate_item.rates[TARGET_MUL_CNT][new_ind];
                now_infos[i].push_back(info);
            }
        }

        cartesian_product(now_infos, [&](const vector<ImmediateInfo> &comb) {
            int last_k = result.indices[comb_size - 1];
            auto &last_item = rate_items[last_k];

            double sum_vol = 0.0;
            for(auto info : comb) {
                sum_vol += info.vol;
            }
            double search_vol = 1.0 - sum_vol;

            bool is_add = search_vol > last_item.init_vol;
            int it_ind = -1;
            if(!is_add) {
                auto it = lower_bound(ALL(last_item.vols[TARGET_MUL_CNT]), search_vol);
                it_ind = distance(last_item.vols[TARGET_MUL_CNT].begin(), it);
            } else {
                auto it = lower_bound(ALL(last_item.post_add_vols[TARGET_MUL_CNT]), search_vol);
                it_ind = distance(last_item.post_add_vols[TARGET_MUL_CNT].begin(), it);
            }

            if(it_ind >= (int)last_item.rates[TARGET_MUL_CNT].size()) {
                // 1g未満のペイントは無視
            } else {
                ImmediateInfo last_info;
                last_info.k = last_k;
                last_info.is_add = is_add;
                last_info.vol = is_add ? last_item.post_add_vols[TARGET_MUL_CNT][it_ind] : last_item.vols[TARGET_MUL_CNT][it_ind];
                last_info.fractors = last_item.fractors[TARGET_MUL_CNT][it_ind];
                last_info.rate = last_item.rates[TARGET_MUL_CNT][it_ind];

                vector<ImmediateInfo> new_info = comb;
                new_info.push_back(last_info);
                double cost = eval_cost(input, state, new_info);
                if(cost < best_cost) {
                    best_cost = cost;
                    best_info = new_info;
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

    // int add_cnt = 0;
    for(auto &info : best_info) {
        int now_partition_pos = primitive_group_info.partition_positions[info.k];
        int frac_size = info.fractors.size();

        // 追加は最初にやっておく
        if(info.is_add) {
            action_result.pre_actions.push_back(Action::Add(0, info.k, info.k));
        }

        auto &first_fractor = info.fractors[0];
        if(first_fractor.first == 1 && first_fractor.second == 1) {
            // 全開放
            assert(frac_size == 1);
            action_result.release_actions.push_back(Action::Toggle(now_partition_pos, info.k, now_partition_pos + 1, info.k));
            action_result.post_actions.push_back(Action::Toggle(INIT_PARTITION_POS, info.k, INIT_PARTITION_POS + 1, info.k));
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
                    action_result.pre_actions.push_back(Action::Toggle(stop_par_pos - 1, info.k, stop_par_pos, info.k));
                    action_result.pre_actions.push_back(Action::Toggle(lower_partition - 1, info.k, lower_partition, info.k));
                }
                // 分子の位置で止める
                action_result.pre_actions.push_back(Action::Toggle(release_par_pos - 1, info.k, release_par_pos, info.k));

                if(fi == frac_size - 1) {
                    // 分母の位置で解放する
                    action_result.release_actions.push_back(Action::Toggle(stop_par_pos - 1, info.k, stop_par_pos, info.k));
                    // 最後の仕切り位置は、release地点になる
                    new_group_info.partition_positions[info.k] = release_par_pos - 1;
                } else {
                    // 仮止めした仕切りは解放しておく必要がある
                    action_result.post_actions.push_back(Action::Toggle(release_par_pos - 1, info.k, release_par_pos, info.k));
                }

                upper_partition = release_par_pos;
                lower_partition = stop_par_pos;
            }
            if(new_group_info.partition_positions[info.k] == 0) {
                // 仕切りが小さくなったら復活しておく
                int now_pos = new_group_info.partition_positions[info.k];
                action_result.post_actions.push_back(Action::Toggle(INIT_PARTITION_POS, info.k, INIT_PARTITION_POS + 1, info.k));
                action_result.post_actions.push_back(Action::Toggle(now_pos, info.k, now_pos + 1, info.k));
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
        double pred_turn = comb_size * 4.0 + 2.0; // 1色あたり4ターン + 2.0ターンのバッファ
        if(pred_turn <= obj_turn) {
            results.push_back(result);
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
    // int remain_turn = state.input.T - state.turn - 30; // 30ターンは余裕を持たせる
    // int obj_turn = round((double)remain_turn / (double)(state.input.H - state.deliver_cnt));

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
            if(h % 20 == 0) print_info(state);

            int remain_turn = input.T - state.turn - BUFFER_TURN;
            double obj_turn = (double)remain_turn / (double)(input.H - state.deliver_cnt);

            if(obj_turn >= SWITH_POLICY_OBJ_TURN) {
                auto pre_err = state.error;
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

                // for(int k : range(input.K)) {
                //     int x = primitive_group_info.partition_positions[k];
                //     assert(state.wall.wall_v[x][k]);
                //     cpp_dump(k, x);
                // }

                // tmp
                auto post_err = state.error;
                // if(post_err - pre_err > 0.05) {
                //     cerr << boost::format("<BigError: %f -> %f (H: %d)>") % pre_err % post_err % h << endl;
                // }

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
#include "hpp/color_mixer.hpp"
#include "hpp/common.hpp"
#include "hpp/game.hpp"
#include "hpp/io.hpp"
#include "hpp/utils.hpp"

// ============================================================================
// 定義
// ============================================================================

const double MAX_TIME = 2800.0;
const int INIT_PARTITION_POS = 0; // パーティション初期値
long long MAX_SIMULATE_CNT = 2e7; // 分数パターンの最大数（目安）
const int BUFFER_TURN = 10;       // 念のためバッファを持たせる
const int SEARCH_LEFT = -1;       // 直積の左側を探索
const int SEARCH_RIGHT = 1;       // 直積の右側を探索
const int MAX_SEARCH_NUM = 25;
// const int MIN_SEARCH_NUM = 15;

// ============================================================================
// Main
// ============================================================================

int calc_pred_fractor_turn(int comb_size) {
    return comb_size * 4 + 3; // 追加,配達,削除
}

int calc_pred_greedy_turn(int comb_size) {
    return comb_size * 2;
}
class ColorGroupManager {
  private:
    struct GroupInfo {
        int k;
        int row_num;
        int start_x;
        std::vector<std::pair<int, int>> roots;
        int now_pos;
        int size;
    };

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
    ColorGroupManager(int n_, int k_, int original_k_, int init_pos_ = 2) : n(n_), k(k_), original_k(original_k_), init_pos(init_pos_) {
        infos = construct_group_info();
    }

    vector<int> get_unique_sizes() {
        set<int> unique_denoms;
        for(int ki : range(this->k)) {
            unique_denoms.insert(this->get_size(ki));
        }
        vector<int> denoms(ALL(unique_denoms));
        return denoms;
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

    void apply_reserved_changes(vector<pair<int, int>> &reserved_changes) {
        for(auto &[k_index, pos] : reserved_changes) {
            this->change_now_pos(k_index, pos);
        }
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

        // ルート内の仕切りを外す
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

class FractorManager {
    using KEY = tuple<int, int, int>; // (init_pos, max_denom, apply_frac_cnt)

  private:
    unordered_map<KEY, vector<Fractors>> fractor_map;
    unordered_map<KEY, vector<double>> rates_map;

  public:
    FractorManager(vector<int> &max_denominators) {
        for(auto &max_denom : max_denominators) {
            this->construct(max_denom, 1);
            this->construct(max_denom, 2);
        }
    }

    double calc_rate(const Fractors &fractors) const {
        double rate = 1.0;
        for(auto &fractor : fractors) {
            if(fractor.first == -1) {
                rate = 0.0;
            } else {
                rate *= (double)(fractor.first) / (double)(fractor.second);
            }
        }
        return rate;
    }

    void construct(int max_denom, int apply_frac_cnt) {
        vector<Fractors> fractors;
        vector<double> rates;

        // 全開放
        fractors.push_back({make_pair(1, 1)});
        rates.push_back(1.0);

        // 何もしない
        fractors.push_back({make_pair(-1, -1)});
        rates.push_back(0.0);

        // !INFO
        // 分数の適応パターンが多すぎる場合、パターンの列挙だけでTLEする可能性がある。
        // 仕方なくパターン数を減らすが、もっと良い方法があるかもしれない。

        int MAX_STEP = 1;
        long long simulate_cnt = pow(max_denom, 5);
        if(simulate_cnt > MAX_SIMULATE_CNT) {
            double div = (double)simulate_cnt / (double)MAX_SIMULATE_CNT;
            MAX_STEP = max(1, int(round(pow(div, 1.0 / 5.0))));
        }

        unordered_set<Fractor> fractor_set;
        for(int init_pos : range(max_denom, -1, -1)) {
            for(int fractor_cnt : range(apply_frac_cnt)) {
                // 分数1回適応
                if(fractor_cnt == 0) {
                    for(int denominator : range(max(2, init_pos), max_denom + 1)) {
                        for(int numerator : range(1, denominator)) {
                            Fractor fractor = make_pair(numerator, denominator);
                            Fractor reduced_fractor = reduce_fraction(fractor);
                            if(fractor_set.contains(reduced_fractor)) continue;
                            fractor_set.insert(reduced_fractor);
                            fractors.push_back({fractor});
                            rates.emplace_back(calc_rate(fractors.back()));
                        }
                    }
                } else if(fractor_cnt == 1) {
                    // 分数2回適応
                    for(int d1 : range(max(2, init_pos + 1), max_denom + 1, MAX_STEP)) {
                        for(int n1 : range(1, d1, MAX_STEP)) {
                            Fractor f1 = make_pair(n1, d1);
                            // 次の分母の最小値 = 下側のブロック数 = 前の分子
                            // 次の分母の最大値 += 最大ブロック数 - 前の分母
                            for(int d2 : range(max(2, n1), n1 + (max_denom - d1) + 1, MAX_STEP)) {
                                for(int n2 : range(1, d2, MAX_STEP)) {
                                    Fractor f2 = make_pair(n2, d2);
                                    Fractor fractor = mul_fracs({f1, f2});
                                    if(fractor_set.contains(fractor)) continue;
                                    fractor_set.insert(fractor);
                                    fractors.push_back({f1, f2});
                                    rates.emplace_back(calc_rate(fractors.back()));
                                }
                            }
                        }
                    }
                }
            }

            // 2分探索のためソートしておく必要がある
            auto inds = make_sorted_indices(rates);
            reorder_vector(fractors, inds);
            reorder_vector(rates, inds);

            KEY key = make_tuple(init_pos, max_denom, apply_frac_cnt);
            fractor_map[key] = fractors;
            rates_map[key] = rates;
        }
    }

    const vector<Fractors> &get_fractors(int pos, int max_denom, int apply_frac_cnt) const {
        KEY key = make_tuple(pos, max_denom, apply_frac_cnt);
        return fractor_map.at(key);
    }

    const vector<double> &get_rates(int pos, int max_denom, int apply_frac_cnt) const {
        KEY key = make_tuple(pos, max_denom, apply_frac_cnt);
        return rates_map.at(key);
    }

    pair<double, Fractors> get(int pos, int max_denom, int apply_frac_cnt, int i) const {
        KEY key = make_tuple(pos, max_denom, apply_frac_cnt);
        auto &frac = fractor_map.at(key)[i];
        auto &rate = rates_map.at(key)[i];
        return {rate, frac};
    }
};

struct DicisionAction {
    vector<Action> pre_actions;
    vector<Action> release_actions;
    vector<Action> post_actions;
    int act_cnt;
    int change_color_num;

    vector<pair<int, int>> reserved_changes; // (k_index, pos)

    double cost;
};

class Planner {
  public:
    const int MAX_TURN = 19005; // (4 * 4 + 3 = 19) * 1000 = 19000
    struct PolicyItem {
        int policy_id; // 0: greedy, 1: fract
        int comb_size;
        int limit_turn; // このターン数までに抑えること
    };

    struct TmpResult {
        double cost;
        int pred_turn;
        int policy_id; // 0: greedy, 1: fract
        vector<int> indices;
        vector<double> weights;
    };

    Input &input;
    State &state;
    ColorMixer &mixer;
    vector<vector<TmpResult>> all_tmp_results;
    vector<int> planning_result_indices;
    vector<int> predicted_accumulated_turns;

    Planner(Input &input_, State &state_, ColorMixer &mixer_) : input(input_), state(state_), mixer(mixer_) {
        for(int h : range(input.H)) {
            vector<TmpResult> tmp_results;
            // PolicyGreedy
            for(int comb_size : range(1, min(5, input.K) + 1)) {
                auto r = mixer.get_greedy_result(h, comb_size);
                TmpResult tmp_result{.cost = r.cost,
                                     .pred_turn = calc_pred_greedy_turn(comb_size),
                                     .policy_id = 0, // 0: greedy
                                     .indices = r.indices,
                                     .weights = r.weights};
                assert(comb_size >= (int)tmp_result.indices.size());
                tmp_results.push_back(tmp_result);
            }
            // PolicyFractor
            for(int comb_size : range(2, 5)) {
                auto results = mixer.get_fract_results(h, comb_size);
                auto r = results[0]; // 先頭がBest
                TmpResult tmp_result{.cost = r.cost,
                                     .pred_turn = calc_pred_fractor_turn(comb_size),
                                     .policy_id = 1, // 1: fractor
                                     .indices = r.indices,
                                     .weights = r.weights};
                assert(comb_size >= (int)tmp_result.indices.size());
                tmp_results.push_back(tmp_result);
            }
            all_tmp_results.push_back(tmp_results);
        }
        planning();
    }

    PolicyItem get_policy(int h) {
        assert(0 <= h && h < input.H);
        int best_index = planning_result_indices[h];
        double best_cost = all_tmp_results[h][best_index].cost;
        auto limit_turn = predicted_accumulated_turns[h];
        for(int i : range(all_tmp_results[h].size())) {
            if(i == best_index) continue;
            auto &temp_result = all_tmp_results[h][i];
            if(temp_result.cost < best_cost && temp_result.pred_turn + state.turn <= limit_turn) {
                best_index = i;
                best_cost = temp_result.cost;
            }
        }
        auto &best_result = all_tmp_results[h][best_index];
        PolicyItem policy = {.policy_id = best_result.policy_id, .comb_size = (int)best_result.indices.size(), .limit_turn = limit_turn};
        return policy;
    }

    void planning() {
        int buf_max_turn = input.T - BUFFER_TURN;
        int max_trun = min(buf_max_turn, MAX_TURN);
        vector<vector<pair<double, int>>> dp(input.H + 1, vector<pair<double, int>>(max_trun + 1, {1e18, -1}));
        dp[0][0] = {0.0, -1};

        // DP計算による各ターンの最適戦略を求める
        // !! 全体でO(10^8)程度
        for(int h : range(input.H)) {
            for(int t : range(max_trun + 1)) {
                for(int i : range(all_tmp_results[h].size())) {
                    auto &item = all_tmp_results[h][i];
                    if(item.pred_turn + t <= max_trun) {
                        double new_cost = dp[h][t].first + item.cost;
                        if(new_cost < dp[h + 1][t + item.pred_turn].first) {
                            dp[h + 1][t + item.pred_turn] = {new_cost, i};
                        }
                    }
                }
            }
        }

        // H回目の最適ターン
        int best_turn = -1;
        double best_cost = 1e18;
        for(int t : range(max_trun + 1)) {
            if(dp[input.H][t].first < best_cost) {
                best_cost = dp[input.H][t].first;
                best_turn = t;
            }
        }

        assert(buf_max_turn >= best_turn);
        int remain_turn = buf_max_turn - best_turn;

        // 計画復元
        planning_result_indices.resize(input.H);
        int t = best_turn;
        for(int h = input.H; h > 0; --h) {
            int i = dp[h][t].second;
            planning_result_indices[h - 1] = i;
            t -= all_tmp_results[h - 1][i].pred_turn;
        }

        // 累積ターン数を計算
        predicted_accumulated_turns.resize(input.H);
        int first_index = planning_result_indices[0];
        predicted_accumulated_turns[0] = all_tmp_results[0][first_index].pred_turn;
        for(int h = 1; h < input.H; ++h) {
            int best_index = planning_result_indices[h];
            predicted_accumulated_turns[h] = predicted_accumulated_turns[h - 1] + all_tmp_results[h][best_index].pred_turn;
        }

        // 各hのターン数上限
        for(int h : range(input.H)) {
            double add_turn = (double)remain_turn / (double)input.H * (h + 1);
            predicted_accumulated_turns[h] += add_turn;
        }

        assert(abs(predicted_accumulated_turns[input.H - 1] - buf_max_turn) <= 3);
    }
};

class PolicyFractor {
  public:
    Input &input;
    State &state;
    ColorMixer &mixer;
    ColorGroupManager &color_group_manager;
    FractorManager &fractor_manager;
    TimeKeeper &time_keeper;

    double start_time = 0.0; // TODO
    struct ImmediateInfo {
        int k;
        bool is_add;
        double rate;
        double vol;
        Fractors fractors;
    };

    PolicyFractor(Input &input_, State &state_, ColorMixer &mixer_, ColorGroupManager &color_group_manager_, FractorManager &fractor_manager_,
                  TimeKeeper &time_keeper_)
        : input(input_), state(state_), mixer(mixer_), color_group_manager(color_group_manager_), fractor_manager(fractor_manager_), time_keeper(time_keeper_) {
    }

    tuple<int, int> search_target_weight_idx(int k, double target_vol, bool is_add, int max_mul_cnt) {
        double now_vol = color_group_manager.get_paint(k, this->state).vol;
        int now_pos = color_group_manager.get_now_pos(k);
        int max_group_size = color_group_manager.get_size(k);
        auto &rates = fractor_manager.get_rates(now_pos, max_group_size, max_mul_cnt);

        // ---------------------------------------
        // now_vol * rate = target_vol
        // rate = target_vol / now_vol
        // rate = target_vol / (now_vol + 1.0)
        // ---------------------------------------
        double search_rate;
        if(is_add) {
            search_rate = target_vol / (1.0 + now_vol);
        } else {
            search_rate = target_vol / now_vol;
        }
        auto it = upper_bound(ALL(rates), search_rate);
        int it_ind = distance(rates.begin(), it);
        int rates_size = (int)rates.size();
        if(it_ind >= rates_size) {
            it_ind = rates_size - 1;
        }
        return {it_ind, rates_size};
    }

    double eval_cost(vector<ImmediateInfo> &immeediate_info) {
        auto &now_target = this->state.input.target[this->state.deliver_cnt];

        double sum_vol = 0.0;
        int add_cnt = 0;
        vector<double> vols;
        vector<Color> colors;
        for(auto info : immeediate_info) {
            vols.emplace_back(info.vol);
            colors.emplace_back(this->input.own[info.k]);
            sum_vol += info.vol;
            if(info.is_add) add_cnt++;
        }

        Color mixed_color = mix(vols, colors);
        double err_cost = eval_error(mixed_color, now_target) * 1e4;
        double discard_cost = max(0.0, sum_vol - 1.0) * (double)(this->input.D);

        int total_add_cnt = this->state.add_cnt + add_cnt;
        if(total_add_cnt > input.H) {
            double add_cost = (total_add_cnt - input.H) * (double)(this->input.D);
            return err_cost + add_cost;
        } else {
            return err_cost + discard_cost;
        }
    }

    tuple<vector<ImmediateInfo>, double> eval_one_result(ColorMixer::Result &constrait, vector<int> &max_frac_cnt) {
        int comb_size = constrait.indices.size();

        vector<vector<ImmediateInfo>> infos;
        for(int comb_ind : range(comb_size)) {
            auto &k = constrait.indices[comb_ind];
            auto &target_vol = constrait.weights[comb_ind];
            double now_vol = color_group_manager.get_paint(k, state).vol;
            bool is_add = (target_vol > now_vol) ? true : false;
            auto [it_ind, max_ind] = search_target_weight_idx(k, target_vol, is_add, max_frac_cnt[comb_ind]);
            vector<ImmediateInfo> immediate_infos;

            for(int j : range(SEARCH_LEFT, SEARCH_RIGHT)) {
                if(it_ind + j < 0 || it_ind + j >= max_ind) continue;
                int new_ind = it_ind + j;
                auto [rate, fractors] =
                    fractor_manager.get(color_group_manager.get_now_pos(k), color_group_manager.get_size(k), max_frac_cnt[comb_ind], new_ind);
                double vol;
                if(is_add) {
                    vol = (now_vol + 1.0) * rate;
                } else {
                    vol = now_vol * rate;
                }
                ImmediateInfo info = {.k = k, .is_add = is_add, .rate = rate, .vol = vol, .fractors = fractors};
                immediate_infos.emplace_back(info);
            }
            infos.emplace_back(move(immediate_infos));
        }

        double best_cost = 1e9;
        vector<ImmediateInfo> best_info;
        cartesian_product(infos, [&](vector<ImmediateInfo> &comb) {
            double sum_vol = 0.0;
            for(const auto &info : comb) {
                sum_vol += info.vol;
            }
            if(sum_vol > 1.0 - 1e-6) {
                double cost = eval_cost(comb);
                if(cost < best_cost) {
                    best_cost = cost;
                    best_info = comb;
                }
            }
        });

        return {best_info, best_cost};
    }

    DicisionAction construct_from_immediateinfo(vector<ImmediateInfo> &best_info) {
        DicisionAction action_result;
        action_result.change_color_num = (int)best_info.size();
        vector<pair<int, int>> reserved_changes;

        for(auto &info : best_info) {
            int now_partition_pos = color_group_manager.get_now_pos(info.k);
            int frac_size = info.fractors.size();

            auto &first_fractor = info.fractors[0];
            if(first_fractor.first == -1 && first_fractor.second == -1) {
                // 何もしない
                continue;
            } else if(first_fractor.first == 1 && first_fractor.second == 1) {
                // 全開放
                assert(frac_size == 1);
                if(now_partition_pos != 0) {
                    // 先に仕切りを解放する
                    action_result.release_actions.emplace_back(color_group_manager.get_toggle_action(info.k, now_partition_pos));
                }
                if(info.is_add) {
                    // 絵の具追加する(release_act)
                    action_result.release_actions.emplace_back(color_group_manager.get_add_paint_action(info.k));
                }
                reserved_changes.emplace_back(info.k, 0);
            } else {
                // 分割n回適応
                int upper_partition = 0;
                int lower_partition = now_partition_pos;
                for(int fi : range(frac_size)) {
                    auto &fractor = info.fractors[fi];
                    // 上の仕切りから、分母だけ進んだのがstopしたいしきり位置
                    int stop_par_pos = upper_partition + fractor.second;
                    // stopする仕切りから、分子だけ進んだのが、releaseする仕切り位置
                    int release_par_pos = stop_par_pos - fractor.first;
                    if(stop_par_pos != lower_partition) {
                        // 現在の仕切りを動かす必要があるなら、仕切りを拡張する
                        action_result.pre_actions.emplace_back(color_group_manager.get_toggle_action(info.k, stop_par_pos));
                        if(lower_partition != 0) {
                            // 現在の仕切り位置が0でないなら、元の仕切りを解放しておく
                            action_result.pre_actions.emplace_back(color_group_manager.get_toggle_action(info.k, lower_partition));
                        }
                    }
                    // 拡張した後に追加する
                    if(fi == 0 && info.is_add) {
                        action_result.pre_actions.emplace_back(color_group_manager.get_add_paint_action(info.k));
                    }
                    // 分子の位置で止める
                    action_result.pre_actions.emplace_back(color_group_manager.get_toggle_action(info.k, release_par_pos));

                    if(fi == frac_size - 1) {
                        // 分母の位置で解放する
                        action_result.release_actions.emplace_back(color_group_manager.get_toggle_action(info.k, stop_par_pos));
                        // 最後の仕切り位置は、release地点になる
                        reserved_changes.emplace_back(info.k, release_par_pos);
                    } else {
                        // 仮止めした仕切りは解放しておく必要がある
                        action_result.post_actions.emplace_back(color_group_manager.get_toggle_action(info.k, release_par_pos));
                    }

                    upper_partition = release_par_pos;
                    lower_partition = stop_par_pos;
                }
            }
        }
        int act_cnt = action_result.pre_actions.size() + action_result.release_actions.size() + action_result.post_actions.size();
        action_result.act_cnt = act_cnt;
        action_result.reserved_changes = reserved_changes;

        return action_result;
    }

    DicisionAction dicision_action(Planner::PolicyItem &policy_item) {
        // TODO 時間に応じてMAX_SEARCH_NUMを調整したい
        // if(this->start_time == 0.0) {
        //     this->start_time = this->time_keeper.getElapsedTime();
        // }
        // double elapsed_time = this->time_keeper.getElapsedTime() - this->start_time;

        double best_cost = 1e9;
        vector<ImmediateInfo> best_info;
        auto results = mixer.get_fract_results(state.deliver_cnt, policy_item.comb_size);
        int remain_turn = policy_item.limit_turn - state.turn - calc_pred_fractor_turn(policy_item.comb_size);
        // assert(remain_turn >= 0); // Addが多いと残りターンがマイナスになる可能性がある

        for(int i : range(min((int)results.size(), MAX_SEARCH_NUM))) {
            auto &result = results[i];
            vector<int> max_frac_cnt(policy_item.comb_size, 1);
            if(policy_item.comb_size == 4) {
                // 4色配合の場合、分数を2回適応できる可能性がある
                int max_double_frac_num = min(4, (int)(remain_turn / 4.0));
                if(max_double_frac_num > 0) {
                    for(int i : range(max_double_frac_num)) {
                        max_frac_cnt[policy_item.comb_size - i - 1] = 2;
                    }
                }
            }
            do {
                auto [now_info, now_cost] = this->eval_one_result(result, max_frac_cnt);
                if(now_cost < best_cost) {
                    best_cost = now_cost;
                    best_info = now_info;
                }
            } while(next_permutation(ALL(max_frac_cnt)));
        }

        assert((int)best_info.size() != 0);

        auto action_result = construct_from_immediateinfo(best_info);
        action_result.cost = best_cost;
        return action_result;
    }
};
class PolicyGreedy {
  public:
    Input &input;
    State &state;
    ColorMixer &color_mixer;
    vector<Color> mix_cache;

    PolicyGreedy(Input &input_, State &state_, ColorMixer &color_mixer_) : input(input_), state(state_), color_mixer(color_mixer_) {
    }

    DicisionAction dicision_action(Planner::PolicyItem &policy_item) {
        auto result = color_mixer.get_greedy_result(state.deliver_cnt, policy_item.comb_size);
        vector<Action> actions;
        for(const auto &k : result.indices) {
            actions.push_back(Action::Add(input.N - 1, 0, k));
        }

        int total_add_cnt = state.add_cnt + policy_item.comb_size;

        DicisionAction action_result;
        action_result.pre_actions = actions;
        action_result.cost = result.cost;
        action_result.change_color_num = 99; // TODO: 特に意味がない値
        return action_result;
    }
};

void print_info(State &state) {
    auto [deliver_cost, err_cost, total_cost] = state.get_score();
    cerr << boost::format("H: %4d | Turn: %5d/%5d | Add: %4d | Discard: %4d (%5d loss) | Score: %5d (add: %5d, err: %5d)") % state.deliver_cnt % state.turn %
                state.input.T % state.add_cnt % state.discard_cnt % int(state.discard * 1e4) % total_cost % deliver_cost % err_cost
         << "\n";
}

void apply_actions(DicisionAction &dicision_act, State &state, Input &input, bool is_end) {
    for(const auto &act : dicision_act.pre_actions) {
        state.apply(act);
    }
    for(const auto &act : dicision_act.release_actions) {
        state.apply(act);
    }
    state.apply(Action::Deliver(input.N - 1, 0));

    if(is_end) return; // 最終ターンは配達したら終了

    while(state.get_paint(input.N - 1, 0).vol > 1e-6) {
        state.apply(Action::Discard(input.N - 1, 0));
    }
    for(const auto &act : dicision_act.post_actions) {
        state.apply(act);
    }
}

void solve() {
    TimeKeeper time_keeper(MAX_TIME);

    Input input = parse_input();
    ColorGroupManager color_group_manager(input.N, input.K, input.K, INIT_PARTITION_POS);
    auto unique_sizes_ = color_group_manager.get_unique_sizes();

    FractorManager fractor_manager(unique_sizes_);
    auto init_wall = color_group_manager.struct_init_wall(input);
    State state(init_wall, input);
    ColorMixer mixer(input);

    PolicyGreedy policy_greedy(input, state, mixer);
    PolicyFractor policy_fractor(input, state, mixer, color_group_manager, fractor_manager, time_keeper);
    Planner planner(input, state, mixer);

    int policy_greedy_cnt = 0;
    double policy_err_sum = 0.0;
    map<int, int> act_cnt;
    map<int, int> color_cnt;

    try {
        // Main Loop
        for(int h : range(input.H)) {
            if(h % 10 == 0) print_info(state);
            auto policy = planner.get_policy(h);

            DicisionAction best_act;
            if(policy.policy_id == 0) {
                best_act = policy_greedy.dicision_action(policy);
                policy_greedy_cnt++;
                policy_err_sum += best_act.cost;
            } else {
                best_act = policy_fractor.dicision_action(policy);
            }
            apply_actions(best_act, state, input, (state.deliver_cnt + 1) == input.H);
            color_group_manager.apply_reserved_changes(best_act.reserved_changes);

            act_cnt[best_act.change_color_num] += best_act.act_cnt;
            color_cnt[best_act.change_color_num]++;
        }
        print_info(state);
    } catch(const exception &e) {
        Output output = Output{init_wall, state.actions};
        print_output(output);
        cerr << "Exception: " << e.what() << "\n";
        exit(1);
    }

    // 情報
    if(policy_greedy_cnt > 0) {
        cerr << boost::format("PolicyGreedy %d times. Error %d") % policy_greedy_cnt % (policy_err_sum) << "\n";
    }
    for(const auto &p : act_cnt) {
        int color_num = p.first;
        int call = color_cnt[color_num];
        int total = p.second;
        double avg = (double)total / (double)call;
        cerr << boost::format("color num: %d, call: %d, total: %d, avg: %f") % color_num % call % total % avg << "\n";
    }
    cerr << boost::format("K: %d, T:%d, D:%d") % input.K % input.T % input.D << "\n";
    cerr << boost::format("score: %d, elapsed: %f, turn: %d/%d") % get<2>(state.get_score()) % time_keeper.getElapsedTime() % state.turn % input.T << "\n";

    // output
    Output output = Output{init_wall, state.actions};
    print_output(output);
}

int main() {
    solve();
    return 0;
}
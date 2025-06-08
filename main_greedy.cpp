#include "hpp/color_mixer.hpp"
#include "hpp/common.hpp"
#include "hpp/game.hpp"
#include "hpp/greedy_mixer.hpp"
#include "hpp/io.hpp"
#include "hpp/utils.hpp"

// ============================================================================
// 定義
// ============================================================================

double MAX_TIME = 2800.0;   // 最大実行時間
const int BUFFER_TURN = 10; // 念のためバッファを持たせる

const int GREEDY_COLOR_MAX = 4;
const int GROUP_SIZE = 4;

// ============================================================================
// Solve Greedy
// ============================================================================

class ColorGroupManagerForMinimumTrun {
  public:
    struct GroupInfo {
        int idx;
        int pos_l, pos_r;
        int pos_u, pos_d;
    };

    int GROUP_NUMS;
    Input &input;
    std::vector<GroupInfo> infos;

    void construct_group_info() {
        int idx = 0;
        for(int y : range(0, input.N)) {
            for(int x : range(0, input.N, GROUP_SIZE)) {
                GroupInfo info = {
                    .idx = idx,
                    .pos_l = x,
                    .pos_r = x + GROUP_SIZE,
                    .pos_u = y,
                    .pos_d = y + 1,
                };
                infos.push_back(info);
                idx++;
            }
        }
        GROUP_NUMS = (int)infos.size();
    }

    ColorGroupManagerForMinimumTrun(Input &input_) : input(input_) {
        construct_group_info();
    }

    GroupInfo get_info(int idx) {
        assert(0 <= idx && idx < GROUP_NUMS);
        return infos[idx];
    }

    Action get_add_paint_action(int idx, int k) const {
        assert(0 <= idx && idx < GROUP_NUMS);
        auto &info = infos[idx];
        int y = info.pos_u;
        int x = info.pos_l;
        return Action::Add(y, x, k);
    }

    Action get_deliver_paint_action(int idx) const {
        assert(0 <= idx && idx < GROUP_NUMS);
        auto &info = infos[idx];
        int y = info.pos_u;
        int x = info.pos_l;
        return Action::Deliver(y, x);
    }

    Paint get_paint(int idx, State &state) const {
        assert(0 <= idx && idx < GROUP_NUMS);
        auto &info = infos[idx];
        int y = info.pos_u;
        int x = info.pos_l;
        auto paint = state.get_paint(y, x);
        return paint;
    }

    Wall struct_init_wall() {
        vector<vector<bool>> wall_h(input.N - 1, vector<bool>(input.N, true));
        vector<vector<bool>> wall_v(input.N, vector<bool>(input.N - 1, false));

        for(const auto &info : infos) {
            if(info.pos_l > 0) {
                int y = info.pos_u;
                int x = info.pos_l - 1;
                wall_v[y][x] = true;
            }
        }

        return Wall(wall_h, wall_v);
    }
};

class PolicyGreedyForMinimumTrun {
  public:
    struct ImmediateInfo {
        int idx = -1;
        int discard_cnt;
        vector<int> add_indices;
    };

    Input &input;
    State &state;
    GreedyMixer &greedy_mixer;
    ColorGroupManagerForMinimumTrun &color_group_manager;

    PolicyGreedyForMinimumTrun(Input &input_, State &state_, GreedyMixer &greedy_mixer_, ColorGroupManagerForMinimumTrun &color_group_manager_)
        : input(input_), state(state_), greedy_mixer(greedy_mixer_), color_group_manager(color_group_manager_) {
    }

    double eval_cost(double err, int add_cnt, int discard_cnt) {
        double err_cost = err * 1e4;
        int total_add_cnt = this->state.add_cnt + add_cnt;
        if(total_add_cnt > input.H) {
            double add_cost = (total_add_cnt - input.H) * (double)(this->input.D);
            return err_cost + add_cost;
        } else {
            double discard_cost = (double)(input.D) * (double)(discard_cnt);
            return err_cost + discard_cost;
        }
    }

    double eval_cost(Color &mixed, int add_cnt, int discard_cnt) {
        double err = eval_error(mixed, input.target[state.deliver_cnt]);
        return eval_cost(err, add_cnt, discard_cnt);
    }

    pair<double, ImmediateInfo> roop_idx(int idx, int max_turn) {
        double best_cost = 1e18;
        ImmediateInfo best_info;

        auto paint = color_group_manager.get_paint(idx, state);

        // 追加なし
        if(paint.vol > 1.0 - 1e-6) {
            ImmediateInfo info = {
                .idx = idx,
                .discard_cnt = 0,
                .add_indices = {},
            };
            double cost = eval_cost(paint.color, 0, 0);
            if(cost < best_cost) {
                best_cost = cost;
                best_info = info;
            }
        }

        int now_vol = (int)paint.vol;

        // 全廃棄する場合（ターン数が足りない場合は全廃棄できない）
        // TODO: ループの外に出す
        if(now_vol < max_turn) {
            int remain_turn = min(max_turn - now_vol, greedy_mixer.get_color_max());
            auto ret = greedy_mixer.get_greedy_result(state.deliver_cnt, remain_turn);
            ImmediateInfo info = {
                .idx = idx,
                .discard_cnt = now_vol,
                .add_indices = ret.indices,
            };
            double cost = this->eval_cost(ret.err, (int)ret.indices.size(), now_vol);
            if(cost < best_cost) {
                best_cost = cost;
                best_info = info;
            }
        }

        // 廃棄と追加の組み合わせ
        int max_discard_cnt = min(max_turn - 1, (int)paint.vol - 1);
        for(int discard_cnt : range(0, max_discard_cnt + 1)) {
            int max_add_cnt = min(max_turn - discard_cnt, greedy_mixer.get_color_max());
            max_add_cnt = min(max_add_cnt, GROUP_SIZE - (int)paint.vol);
            for(int add_cnt : range(1, max_add_cnt + 1)) {
                int max_comb = greedy_mixer.mixed_colors_cache[add_cnt].size();
                for(int comb_ind : range(max_comb)) {
                    vector<double> vols = {paint.vol - discard_cnt};
                    vector<Color> colors = {paint.color};
                    vols.push_back((double)add_cnt);
                    auto &c = greedy_mixer.mixed_colors_cache[add_cnt][comb_ind];
                    colors.push_back(c);
                    auto &inds = greedy_mixer.mixed_colors_inds_cache[add_cnt][comb_ind];
                    Color mixed_color = mix(vols, colors);
                    double cost = eval_cost(mixed_color, add_cnt, discard_cnt);
                    ImmediateInfo info = {
                        .idx = idx,
                        .discard_cnt = discard_cnt,
                        .add_indices = inds,
                    };
                    if(cost < best_cost) {
                        best_cost = cost;
                        best_info = info;
                    }
                }
            }
        }

        return {best_cost, best_info};
    }

    vector<Action> create_action(ImmediateInfo &info) {
        vector<Action> actions;
        auto group_info = color_group_manager.get_info(info.idx);
        int y = group_info.pos_u;
        int x = group_info.pos_l;
        for(int i : range(info.discard_cnt)) {
            actions.push_back(Action::Discard(y, x));
        }
        for(int i : info.add_indices) {
            actions.push_back(Action::Add(y, x, i));
        }
        actions.push_back(Action::Deliver(y, x));
        return actions;
    }

    vector<Action> dicision_action() {
        double best_cost = 1e18;
        ImmediateInfo best_info;

        int remain_deliver = input.H - state.deliver_cnt;
        int remain_turn = input.T - BUFFER_TURN - state.turn;
        double obj_turn = (double)remain_turn / (double)remain_deliver;
        assert(obj_turn > 2.0);

        for(int idx : range(color_group_manager.GROUP_NUMS)) {
            auto [cost, info] = roop_idx(idx, (int)obj_turn - 1); // deliver分は先回りして減らす
            if(cost < best_cost) {
                best_cost = cost;
                best_info = info;
            }
        }

        assert(best_info.idx != -1);
        int pred_turn = best_info.discard_cnt + (int)best_info.add_indices.size() + 1;

        assert(pred_turn <= (int)obj_turn);
        auto actions = create_action(best_info);
        return actions;
    }
};

pair<Output, State> solve_greedy(Input &input) {
    ColorGroupManagerForMinimumTrun color_group_manager_min_turn(input);
    auto init_wall = color_group_manager_min_turn.struct_init_wall();
    State state(init_wall, input);
    GreedyMixer mixer(input, GREEDY_COLOR_MAX);
    PolicyGreedyForMinimumTrun policy_greedy_for_min_turn(input, state, mixer, color_group_manager_min_turn);

    try {
        // Main Loop
        for(int h : range(input.H)) {
            if(h % 10 == 0 || h >= 990) state.print_info();
            auto best_act = policy_greedy_for_min_turn.dicision_action();
            state.apply_actions(best_act);
        }
        state.print_info();
    } catch(const exception &e) {
        Output output = Output{init_wall, state.actions};
        print_output(output);
        cerr << "Exception: " << e.what() << "\n";
        exit(1);
    }

    Output output = Output{init_wall, state.actions};
    return {output, state};
}

// ============================================================================
// 実行
// ============================================================================

void solve() {
    TimeKeeper time_keeper(MAX_TIME);
    Input input = parse_input();
    auto [output, state] = solve_greedy(input);

    cerr << boost::format("K: %d, T:%d, D:%d") % input.K % input.T % input.D << "\n";
    cerr << boost::format("score: %d, elapsed: %f, turn: %d/%d") % get<2>(state.get_score()) % time_keeper.getElapsedTime() % state.turn % input.T << "\n";

    // output
    if(IS_DEBUG) {
        cout << boost::format("%d %d") % get<2>(state.get_score()) % state.turn << "\n";
    } else {
        print_output(output);
    }
}

int main() {
    solve();
    return 0;
}
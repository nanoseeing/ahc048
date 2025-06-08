#include "hpp/color_mixer.hpp"
#include "hpp/common.hpp"
#include "hpp/game.hpp"
#include "hpp/greedy_mixer.hpp"
#include "hpp/io.hpp"
#include "hpp/solver_greedy.hpp"
#include "hpp/solver_opt.hpp"
#include "hpp/utils.hpp"

// ============================================================================
// 共通
// ============================================================================

pair<Output, State> solve_fractor(Input &input, TimeKeeper &time_keeper) {
    ColorGroupManager color_group_manager(input.N, input.K, input.K, INIT_PARTITION_POS);
    auto unique_sizes_ = color_group_manager.get_unique_sizes();

    FractorManager fractor_manager(unique_sizes_);
    auto init_wall = color_group_manager.struct_init_wall(input);
    State state(init_wall, input);
    ColorMixer mixer(input);

    Planner planner(input, state, mixer);
    PolicyGreedy policy_greedy(input, state, mixer);
    PolicyFractor policy_fractor(input, state, mixer, color_group_manager, fractor_manager, time_keeper);

    int policy_greedy_cnt = 0;
    double policy_err_sum = 0.0;
    map<int, int> act_cnt;
    map<int, int> color_cnt;

    try {
        // Main Loop
        for(int h : range(input.H)) {
            if(h % 10 == 0 || h >= 990) state.print_info();
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
        state.print_info();
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

    Output output = Output{init_wall, state.actions};
    return {output, state};
}

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

void solve() {
    TimeKeeper time_keeper(MAX_TIME);
    Input input = parse_input();

    State state;
    Output output;
    if(input.T <= 64000) {
        tie(output, state) = solve_greedy(input);
    } else {
        tie(output, state) = solve_fractor(input, time_keeper);
    }

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
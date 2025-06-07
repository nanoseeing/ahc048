#include <bits/stdc++.h>
using namespace std;
#include <Eigen/Core>
#include <Eigen/Dense>
#include <array>
#include <vector>

#include "hpp/common.hpp"
#include "hpp/ex/nnls.hpp"
#include "hpp/game.hpp"
#include "hpp/io.hpp"
#include "hpp/utils.hpp"

class Planner {
  public:
    struct PolicyItem {
        int turn;
        double cost;
    };

    Input &input;
    vector<PolicyItem> planning_polices;
    vector<int> predicted_accumulated_turns;

    Planner(Input &input_, vector<vector<PolicyItem>> &policy_item) : input(input_) {
        planning(policy_item);
    }

    PolicyItem get_policy(int h) {
        assert(0 <= h && h < input.H);
        return planning_polices[h];
    }

    void planning(vector<vector<PolicyItem>> &policy_items) {
        const int MAX_TURN = 19005; // (4 * 4 + 3 = 19) * 1000 = 19000
        vector<vector<pair<double, int>>> dp(input.H + 1, vector<pair<double, int>>(MAX_TURN + 1, {1e18, -1}));
        for(int h : range(input.H + 1)) {
            dp[h][0] = {0.0, -1};
        }

        // DP計算による各ターンの最適戦略を求める
        // !! 全体でO(10^8)程度
        for(int h : range(input.H)) {
            for(int t : range(MAX_TURN)) {
                for(int i : range(policy_items[h].size())) {
                    auto &item = policy_items[h][i];
                    if(item.turn + t <= MAX_TURN) {
                        double new_cost = dp[h][t].first + item.cost;
                        if(new_cost < dp[h + 1][t + item.turn].first) {
                            dp[h + 1][t + item.turn] = {new_cost, i};
                        }
                    }
                }
            }
        }

        // 計画復元
        int best_turn = -1;
        double best_cost = 1e18;
        for(int t : range(MAX_TURN)) {
            if(dp[input.H][t].first < best_cost) {
                best_cost = dp[input.H][t].first;
                best_turn = t;
            }
        }

        planning_polices.resize(input.H);
        int t = best_turn;
        for(int h = input.H; h > 0; --h) {
            int i = dp[h][t].second;
            planning_polices[h - 1] = policy_items[h - 1][i];
            t -= policy_items[h - 1][i].turn;
        }

        // 累積ターン数を計算
        predicted_accumulated_turns.resize(input.H);
        predicted_accumulated_turns[0] = planning_polices[0].turn;
        for(int h = 1; h < input.H; ++h) {
            predicted_accumulated_turns[h] = predicted_accumulated_turns[h - 1] + planning_polices[h].turn;
        }
    }
};

int main() {
    Input input = parse_input();
    TimeKeeper timer(100.0);

    vector<vector<Planner::PolicyItem>> policy_items(input.H);
    for(int h = 0; h < input.H; ++h) {
        int turn = 15;
        double cost = 1e5;
        for(int i = 0; i < 10; ++i) {
            turn += rand() % 10;              // ランダムにターン数を増やす
            cost += (rand() % 1000) / 1000.0; // ランダムにコストを増やす
            policy_items[h].push_back({turn, cost});
        }
    }

    Planner planner(input, policy_items);
    for(int h = 0; h < input.H; ++h) {
        auto policy = planner.get_policy(h);
        cerr << "Turn: " << h << ", Cost: " << policy.cost << "\n";
    }
    cerr << boost::format("Times: %.2f seconds") % timer.getElapsedTime() << "\n";

    return 0;
}

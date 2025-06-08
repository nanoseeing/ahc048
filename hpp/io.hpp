#pragma once
#include "common.hpp"
#include "game.hpp"

// =========================================================
// IO
// =========================================================

struct Output {
    Wall init_wall;
    vector<Action> actions;
};

Input parse_input() {
    Input input;
    cin >> input.N >> input.K >> input.H >> input.T >> input.D;
    input.own.resize(input.K);
    for(int i = 0; i < input.K; ++i) {
        for(int j = 0; j < 3; ++j) {
            cin >> input.own[i][j];
        }
    }
    input.target.resize(input.H);
    for(int i = 0; i < input.H; ++i) {
        for(int j = 0; j < 3; ++j) {
            cin >> input.target[i][j];
        }
    }
    return input;
}

void print_output(Output &output) {
    const auto &wall = output.init_wall;
    for(int i = 0; i < (int)wall.wall_v.size(); ++i) {
        for(int j = 0; j < (int)wall.wall_v[i].size(); ++j) {
            cout << (wall.wall_v[i][j] ? "1" : "0") << " ";
        }
        cout << "\n";
    }
    for(int i = 0; i < (int)wall.wall_h.size(); ++i) {
        for(int j = 0; j < (int)wall.wall_h[i].size(); ++j) {
            cout << (wall.wall_h[i][j] ? "1" : "0") << " ";
        }
        cout << "\n";
    }

    for(const auto &action : output.actions) {
        cout << action.to_string_output() << "\n";
    }
}

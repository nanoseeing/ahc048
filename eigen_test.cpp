#include "hpp/comb.hpp"
#include "hpp/common.hpp"
#include "hpp/game.hpp"
#include "hpp/io.hpp"
// #include "hpp/nnls_pdm.hpp"
#include "hpp/utils.hpp"

void solve() {
    Input input = parse_input();
    ColorMixer mixer(input.own);

    const int SEARCH_SIZE = 10;

    TimeKeeper timer(100.0);
    for(int h : range(input.H)) {
        Color t = input.target[h];
        vector<double> errs;
        for(int k : range(2, min(5, input.K + 1))) {
            auto results = mixer.find_topN(t, SEARCH_SIZE);
            // auto results = mixer.solve_nnls(t, k, SEARCH_SIZE);
            auto best_ret = results[0];
            double w_sum = accumulate(ALL(best_ret.weights), 0.0);
            double e = best_ret.squared_error * 1e4;
            errs.push_back(e);
        }
        // cpp_dump(h, errs);
    }

    cpp_dump(timer.getElapsedTime());
}

int main() {
    solve();
    return 0;
}
#include "hpp/common.hpp"
// #include "hpp/eigen_comb.hpp"
// #include "hpp/comb.hpp"
// #include "hpp/colormixer.hpp"
#include "hpp/game.hpp"
#include "hpp/io.hpp"
#include "hpp/nnls_pdm.hpp"
#include "hpp/utils.hpp"

void solve() {
    Input input = parse_input();
    ColorMixer mixer(input.own);

    vector<int> indices;
    for(int k : range(input.K)) {
        indices.push_back(k);
    }

    for(int h : range(input.H)) {
        Color t = input.target[h];
        auto result = mixer.solve_nnls_for_indices(indices, t, 1e-12, 10000);
        double w_sum = accumulate(ALL(result.weights), 0.0);
        cpp_dump(w_sum, result.squared_error * 1e4, result.weights, result.indices);
    }

    // for(int h : range(input.H)) {
    //     Color t = input.target[h];
    //     auto results = mixer.find_topN(t, 10);
    //     auto first_result = results[0];
    //     cpp_dump(first_result.squared_error * 1e4); //, first_result.weights, first_result.indices);
    //     // auto [w, err, c_hat] = mixer.solve_nnls_for_indices(indices, t);
    //     // cpp_dump(err * 1e4, w);
    // }
}

int main() {
    solve();
    return 0;
}
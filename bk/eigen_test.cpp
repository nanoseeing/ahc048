#include "hpp/common.hpp"
#include "hpp/game.hpp"
#include "hpp/io.hpp"
#include "hpp/nnls.hpp"
// #include "hpp/nnls_pdm.hpp"
#include "hpp/utils.hpp"

void solve() {
    Input input = parse_input();
    ColorMixer mixer(input.own);

    const int SEARCH_SIZE = 10;
    // map<pair<int, int>, vector<double>> err_map;
    for(int h : range(input.H)) {
        Color t = input.target[h];
        vector<double> errs;
        for(int k : range(2, min(5, input.K + 1))) {
            // auto results = mixer.solve_nnls_nCk(input.K, k, MAX_COMB, t);
            auto results = mixer.solve_nnls(t, k, SEARCH_SIZE);
            auto best_ret = results[0];
            double w_sum = accumulate(ALL(best_ret.weights), 0.0);
            double e = best_ret.err * 1e4;
            errs.push_back(e);
            // cpp_dump(input.K, k, e, w_sum);
            // for(const int x : range(min(20, (int)results.size()))) {
            //     auto result = results[x];
            //     double w_sum = accumulate(ALL(result.weights), 0.0);
            //     double e = result.squared_error * 1e4;
            //     errs.push_back(e);
            //     cpp_dump(input.K, k, e, w_sum, result.iter_cnt);
            // }
            // err_map[{h, k}] = errs;
        }
        cpp_dump(h, errs);

        // if(h % 100 == 0) {
        //     cerr << boost::format("H: %4d, err = %.3f, W = %.6f, Iter = %d\n") % h % e % w_sum % result.iter_cnt;
        // }
    }

    // vector<double> errs;
    // for(int h : range(input.H)) {
    //     Color t = input.target[h];
    //     auto result = mixer.solve_nnls_for_indices(indices, t, 1e-7, 1000);
    //     double w_sum = accumulate(ALL(result.weights), 0.0);
    //     double e = result.squared_error * 1e4;
    //     errs.push_back(e);
    //     // if(h % 100 == 0) {
    //     //     cerr << boost::format("H: %4d, err = %.3f, W = %.6f, Iter = %d\n") % h % e % w_sum % result.iter_cnt;
    //     // }
    // }

    // // 統計
    // double total_err = accumulate(ALL(errs), 0.0);
    // double avg_err = total_err / input.H;
    // double max_err = *max_element(ALL(errs));
    // double min_err = *min_element(ALL(errs));
    // double std_err = 0.0;
    // for(double e : errs) {
    //     std_err += (e - avg_err) * (e - avg_err);
    // }
    // std_err = sqrt(std_err / input.H);
    // cpp_dump(total_err, avg_err, max_err, min_err, std_err);

    // for(auto& e : errs) {
    //     cout << e << " ";
    // }
    // cout << endl;
}

int main() {
    solve();
    return 0;
}
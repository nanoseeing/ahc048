// ============================================================================
// TODO
// ============================================================================
#include "comb.hpp"
#include "common.hpp"
#include "utils.hpp"

// ============================================================================
// 定義
// ============================================================================

const int MAX_RATE = 19;

Xorshift32 x32rng;
Xorshift64 x64rng;

// ============================================================================
// Main
// ============================================================================

Color mix(double v1, Color c1, double v2, Color c2) {
    double sum = v1 + v2;
    if(sum <= 0) return {0.0, 0.0, 0.0};
    return {(v1 * c1[0] + v2 * c2[0]) / sum, (v1 * c1[1] + v2 * c2[1]) / sum, (v1 * c1[2] + v2 * c2[2]) / sum};
}

Color mix(vector<double> &vols, vector<Color> &colors) {
    double sum = 0.0;
    Color result = {0.0, 0.0, 0.0};
    for(int i : range(vols.size())) {
        sum += vols[i];
        result[0] += vols[i] * colors[i][0];
        result[1] += vols[i] * colors[i][1];
        result[2] += vols[i] * colors[i][2];
    }
    if(sum <= 0) return {0.0, 0.0, 0.0};
    return {result[0] / sum, result[1] / sum, result[2] / sum};
}

class Wall {
  public:
    vector<vector<bool>> wall_h;
    vector<vector<bool>> wall_v;
    Wall() = default;
    Wall(const vector<vector<bool>> &wall_h, const vector<vector<bool>> &wall_v) {
        // check size
        int horizontal_h = wall_h.size();
        int horizontal_w = wall_h[0].size();
        int vertical_h = wall_v.size();
        int vertical_w = wall_v[0].size();
        assert(horizontal_h + 1 == horizontal_w);
        assert(vertical_h == vertical_w + 1);
        assert(horizontal_h == vertical_w);

        this->wall_h = wall_h;
        this->wall_v = wall_v;
    }

    void switch_h(int i, int j) {
        wall_h[i][j] = wall_h[i][j] ^ true;
    }

    void switch_v(int i, int j) {
        wall_v[i][j] = wall_v[i][j] ^ true;
    }
};

enum class ActionType {
    Add = 1,
    Deliver = 2,
    Discard = 3,
    Toggle = 4,
};

struct Action {
    ActionType type;
    int i, j, k;
    int i2, j2;

    static Action Add(int i, int j, int k) {
        return {ActionType::Add, i, j, k, 0, 0};
    }
    static Action Deliver(int i, int j) {
        return {ActionType::Deliver, i, j, 0, 0, 0};
    }
    static Action Discard(int i, int j) {
        return {ActionType::Discard, i, j, 0, 0, 0};
    }
    static Action Toggle(int i1, int j1, int i2, int j2) {
        return {ActionType::Toggle, i1, j1, 0, i2, j2};
    }

    string to_string() const {
        if(type == ActionType::Add) {
            return boost::str(boost::format("Add: (%d, %d, %d)") % i % j % k);
        } else if(type == ActionType::Deliver) {
            return boost::str(boost::format("Deliver: (%d, %d)") % i % j);
        } else if(type == ActionType::Discard) {
            return boost::str(boost::format("Discard: (%d, %d)") % i % j);
        } else if(type == ActionType::Toggle) {
            return boost::str(boost::format("Toggle: (%d, %d) <-> (%d, %d)") % i % j % i2 % j2);
        } else {
            cerr << "Unknown ActionType!" << endl;
            exit(1);
        }
    }

    string to_string_output() const {
        int typei = static_cast<int>(this->type);
        if(type == ActionType::Add) {
            return boost::str(boost::format("%d %d %d %d") % typei % i % j % k);
        } else if(type == ActionType::Deliver) {
            return boost::str(boost::format("%d %d %d") % typei % i % j);
        } else if(type == ActionType::Discard) {
            return boost::str(boost::format("%d %d %d") % typei % i % j);
        } else if(type == ActionType::Toggle) {
            return boost::str(boost::format("%d %d %d %d %d") % typei % i % j % i2 % j2);
        } else {
            cerr << "Unknown ActionType!" << endl;
            exit(1);
        }
    }
};

struct Input {
    int N, K, H, T, D;
    vector<Color> own;
    vector<Color> target;
};

struct Output {
    Wall init_wall;
    vector<Action> actions;

    void print_output() {
        const auto &wall = init_wall;
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

        for(const auto &action : actions) {
            cout << action.to_string_output() << "\n";
        }
    }
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

Wall struct_init_wall(Input &input_data) {
    vector<vector<bool>> wall_h(input_data.N - 1, vector<bool>(input_data.N, false));
    vector<vector<bool>> wall_v(input_data.N, vector<bool>(input_data.N - 1, false));

    for(int x : range(input_data.N - 1)) {
        for(int y : range(input_data.N - 1)) {
            wall_v[y][x] = true;
        }
    }
    for(int x : range(input_data.N)) {
        wall_h[input_data.N - 1 - 1][x] = true;
    }

    return Wall(wall_h, wall_v);
}

tuple<int, vector<vector<int>>, vector<int>> get_ids(Wall &wall) {
    // TODO 壁の差分だけを更新するようにしたい。
    int N = wall.wall_v.size();
    vector<vector<int>> ids(N, vector<int>(N, -1));
    int ID = 0;
    vector<int> caps;

    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            if(ids[i][j] != -1) continue;

            vector<pair<int, int>> stack = {{i, j}};
            ids[i][j] = ID;
            int cap = 0;

            while(!stack.empty()) {
                auto [ci, cj] = stack.back();
                stack.pop_back();
                cap++;

                if(cj + 1 < N && !wall.wall_v[ci][cj] && ids[ci][cj + 1] == -1) {
                    ids[ci][cj + 1] = ID;
                    stack.emplace_back(ci, cj + 1);
                }
                if(ci + 1 < N && !wall.wall_h[ci][cj] && ids[ci + 1][cj] == -1) {
                    ids[ci + 1][cj] = ID;
                    stack.emplace_back(ci + 1, cj);
                }
                if(cj > 0 && !wall.wall_v[ci][cj - 1] && ids[ci][cj - 1] == -1) {
                    ids[ci][cj - 1] = ID;
                    stack.emplace_back(ci, cj - 1);
                }
                if(ci > 0 && !wall.wall_h[ci - 1][cj] && ids[ci - 1][cj] == -1) {
                    ids[ci - 1][cj] = ID;
                    stack.emplace_back(ci - 1, cj);
                }
            }

            caps.push_back(cap);
            ID++;
        }
    }

    return {ID, ids, caps};
}

struct Paint {
    int id;
    int cap;
    double vol;
    Color color;
};

double eval_error(Color col, Color tgt) {
    return sqrt(pow(col[0] - tgt[0], 2) + pow(col[1] - tgt[1], 2) + pow(col[2] - tgt[2], 2));
}

struct State {
    Input input;
    Wall wall;
    vector<vector<int>> ids;
    vector<Paint> paints;

    vector<Color> delivered;
    vector<Action> actions;

    int turn = 0;
    int add_cnt = 0;
    double error = 0.0;
    int deliver_cnt = 0;

    State(const Wall &init_wall, const Input &input) {
        this->input = input;
        this->wall = init_wall;
        auto [ID, ids, caps] = get_ids(wall);
        this->ids = ids;
        for(int id : range(ID)) {
            this->paints.push_back({id, caps[id], 0.0, {0.0, 0.0, 0.0}});
        }
    }

    tuple<int, int, int> get_score() const {
        int deliver_cost = input.D * max(0, this->add_cnt - deliver_cnt);
        int err_cost = (int)round(1e4 * this->error);
        int total_cost = 1 + deliver_cost + err_cost;
        return {deliver_cost, err_cost, total_cost};
    }

    Paint get_paint(int i, int j) const {
        int id = this->ids[i][j];
        return this->paints[id];
    }

    Paint get_paint(int id) const {
        return this->paints[id];
    }

    void apply_add_less_1gram() {
        // 1gram未満なら追加
        for(int x : range(input.K)) {
            Paint p = get_paint(0, x);
            if(p.vol < 1.0) {
                this->apply(Action::Add(0, x, x));
            }
        }
    }

    void apply_color_rates(vector<int> &rates) {
        // 色を分割する
        vector<Action> div_actions;
        for(int k : range(rates.size())) {
            if(rates[k] == 0 || rates[k] == MAX_RATE) continue;
            Action act = Action::Toggle(MAX_RATE - 1 - rates[k], k, MAX_RATE - rates[k], k);
            div_actions.push_back(act);
            this->apply(act);
        }

        // 色を合成する
        vector<Action> mix_actions;
        for(int k : range(rates.size())) {
            if(rates[k] == 0) continue;
            Action act = Action::Toggle(MAX_RATE - 1, k, MAX_RATE, k);
            mix_actions.push_back(act);
            this->apply(act);
        }

        // 配達 & 廃棄
        this->apply(Action::Deliver(input.N - 1, 0));
        while(get_paint(input.N - 1, 0).vol > 1e-6) {
            this->apply(Action::Discard(input.N - 1, 0));
        }

        // 合成した仕切りを戻す
        for(const auto &act : mix_actions) {
            this->apply(act);
        }

        // 分割した仕切りを戻す
        for(const auto &act : div_actions) {
            this->apply(act);
        }
    }

    void apply_add(const Action &action) {
        this->add_cnt++;
        int id = this->ids[action.i][action.j];
        double w = static_cast<double>(this->paints[id].cap) - this->paints[id].vol;
        if(w <= 1.0) {
            this->paints[id].color = mix(this->paints[id].vol, this->paints[id].color, w, input.own[action.k]);
            this->paints[id].vol = static_cast<double>(this->paints[id].cap);
        } else {
            this->paints[id].color = mix(this->paints[id].vol, this->paints[id].color, 1.0, input.own[action.k]);
            this->paints[id].vol += 1.0;
        }
    }

    void apply_deliver(const Action &action) {
        this->deliver_cnt++;
        int id = this->ids[action.i][action.j];
        if((int)this->delivered.size() >= input.H) {
            cerr << "Error: Too many deliveries." << endl;
            exit(1);
        };
        if(this->paints[id].vol < 1.0 - 1e-6) {
            cerr << "Error: Not enough paint to deliver." << endl;
            exit(1);
        };
        Color col = this->paints[id].color;
        Color tgt = input.target[this->delivered.size()];
        this->error += eval_error(col, tgt);
        this->paints[id].vol = max(0.0, this->paints[id].vol - 1.0);
        this->delivered.push_back(col);
    }

    void apply_discard(const Action &action) {
        int id = this->ids[action.i][action.j];
        this->paints[id].vol = max(0.0, this->paints[id].vol - 1.0);
    }

    void apply_toggle(const Action &action) {
        int i1 = action.i, j1 = action.j;
        int i2 = action.i2, j2 = action.j2;
        if(i1 == i2) {
            auto i = i1;
            auto j = min(j1, j2);
            this->wall.switch_v(i, j);
        } else {
            auto i = min(i1, i2);
            auto j = j1;
            this->wall.switch_h(i, j);
        }
        auto [ID, ids, caps] = get_ids(this->wall);
        if(this->ids[i1][j1] == this->ids[i2][j2] && ids[i1][j1] != ids[i2][j2]) {
            auto id1 = ids[i1][j1];
            auto id2 = ids[i2][j2];
            auto v = this->paints[this->ids[i1][j1]].vol;
            auto vols = vector<double>(ID, 0.0);
            auto colors = vector<Color>(ID, {0.0, 0.0, 0.0});
            for(int i : range(input.N)) {
                for(int j : range(input.N)) {
                    vols[ids[i][j]] = this->paints[this->ids[i][j]].vol;
                    colors[ids[i][j]] = this->paints[this->ids[i][j]].color;
                }
            }
            vols[id1] = v * (double)caps[id1] / (double)(caps[id1] + caps[id2]);
            vols[id2] = v * (double)caps[id2] / (double)(caps[id1] + caps[id2]);
            this->ids = ids;
            vector<Paint> new_paints(ID);
            for(int id : range(ID)) {
                new_paints[id] = {id, caps[id], vols[id], colors[id]};
            }
            this->paints = new_paints;
        } else if(this->ids[i1][j1] != this->ids[i2][j2] && ids[i1][j1] == ids[i2][j2]) {
            auto id = ids[i1][j1];
            auto id1 = this->ids[i1][j1];
            auto id2 = this->ids[i2][j2];
            auto v1 = this->paints[id1].vol;
            auto v2 = this->paints[id2].vol;
            auto c1 = this->paints[id1].color;
            auto c2 = this->paints[id2].color;
            auto vols = vector<double>(ID, 0.0);
            auto colors = vector<Color>(ID, {0.0, 0.0, 0.0});
            for(int i : range(input.N)) {
                for(int j : range(input.N)) {
                    vols[ids[i][j]] = this->paints[this->ids[i][j]].vol;
                    colors[ids[i][j]] = this->paints[this->ids[i][j]].color;
                }
            }
            vols[id] = v1 + v2;
            colors[id] = mix(v1, c1, v2, c2);
            this->ids = ids;
            vector<Paint> new_paints(ID);
            for(int id : range(ID)) {
                new_paints[id] = {id, caps[id], vols[id], colors[id]};
            }
            this->paints = new_paints;
        }
    }

    void apply(const Action &action) {
        if(turn >= input.T) {
            cerr << "Error: Too many turns." << endl;
            exit(1);
        }

        this->turn++;
        this->actions.push_back(action);

        if(action.type == ActionType::Add) {
            this->apply_add(action);
        } else if(action.type == ActionType::Deliver) {
            this->apply_deliver(action);
        } else if(action.type == ActionType::Discard) {
            this->apply_discard(action);
        } else if(action.type == ActionType::Toggle) {
            this->apply_toggle(action);
        } else {
            cerr << "Unknown action type: " << (int)action.type << endl;
            exit(1);
        }
    }

    void debug() {
        for(const auto &paint : this->paints) {
            if(paint.vol < 1e-6) continue; // 1g未満は表示しない
            cerr << boost::format("ID: %d, Cap: %d, Vol: %.2f, Color: (%.2f, %.2f, %.2f)") % paint.id % paint.cap %
                        paint.vol % paint.color[0] % paint.color[1] % paint.color[2]
                 << endl;
        }
    }
};

double eval_cost(State &state, const vector<int> &rates) {
    auto &now_target = state.input.target[state.deliver_cnt];

    double sum_vol = 0.0;
    vector<double> vols;
    vector<Color> colors;
    for(int k : range(state.input.K)) {
        Paint paint = state.get_paint(0, k);
        double new_vol = paint.vol * (double)(rates[k]) / (double)MAX_RATE;
        vols.push_back(new_vol);
        colors.push_back(paint.color);
        sum_vol += new_vol;
    }

    Color mixed_color = mix(vols, colors);
    double err_cost = eval_error(mixed_color, now_target) * 1e4;
    double discard_cost = max(0.0, sum_vol - 1.0) * (double)(state.input.D);

    return err_cost + discard_cost;
}

void PolicyGreedy(State &state) {
    auto target_color = state.input.target[state.deliver_cnt];

    double min_cost = 1e9;
    int min_k = -1;
    bool is_add = false;
    for(int k : range(state.input.K)) {
        Paint paint = state.get_paint(0, k);
        double now_cost = eval_error(paint.color, target_color) * 1e4;
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

vector<int> get_rates(Input &input, State &state, ColorMixer &mixer, int obj_turn) {
    const int TOP_N = 10000;
    const int MAX_RESULT = 1000;

    Color target = input.target[state.deliver_cnt];
    auto all_results = mixer.find_topN(target, TOP_N);

    vector<ColorMixer::Result> results;
    for(const auto &result : all_results) {
        int comb_size = result.indices.size();
        if(obj_turn >= 17) {
            // 4色OK
            results.push_back(result);
        } else if(13 <= obj_turn && obj_turn < 17 && comb_size <= 3) {
            // 3色まで
            results.push_back(result);
        } else if(9 <= obj_turn && obj_turn < 13 && comb_size == 2) {
            // 2色まで
            results.push_back(result);
        }
        if((int)results.size() >= MAX_RESULT) {
            break;
        }
    }

    vector<vector<double>> all_sorted_vols(input.K);
    for(int k : range(input.K)) {
        Paint paint = state.get_paint(0, k);
        for(int r : range(MAX_RATE + 1)) {
            double vol = paint.vol * (double)(r) / (double)(MAX_RATE);
            all_sorted_vols[k].push_back(vol);
        }
    }

    double best_cost = 1e9;
    vector<int> best_rates;

    for(const auto &result : results) {
        int comb_size = result.indices.size();

        // weightsに近いvolumeを持つペイントのrateを求める(最後のペイントは無視)
        vector<vector<int>> target_rates(comb_size - 1);
        for(int i : range(comb_size - 1)) {
            int k = result.indices[i];
            double weight = result.weights[i];
            auto it = upper_bound(ALL(all_sorted_vols[k]), weight);
            int it_ind = distance(all_sorted_vols[k].begin(), it);
            if(it_ind >= MAX_RATE) {
                target_rates[i].push_back(MAX_RATE);
                target_rates[i].push_back(MAX_RATE - 1);
                // target_rates[i].push_back(MAX_RATE - 2);
            } else if(it == all_sorted_vols[k].begin()) {
                target_rates[i].push_back(it_ind);
                target_rates[i].push_back(it_ind + 1);
                // target_rates[i].push_back(it_ind + 2);
            } else {
                target_rates[i].push_back(it_ind - 1);
                target_rates[i].push_back(it_ind);
                // target_rates[i].push_back(it_ind + 1);
            }
        }

        // target_ratesの直積を試す
        cartesian_product(target_rates, [&](const vector<int> &comb) {
            vector<int> temp_rates(input.K, 0);
            double sum_vol = 0.0;
            for(int i : range(comb_size - 1)) {
                int k = result.indices[i];
                int rate = comb[i];
                sum_vol += all_sorted_vols[k][rate];
                temp_rates[k] = rate;
            }
            int last_k = result.indices[comb_size - 1];
            double search_vol = 1.0 - sum_vol;
            auto it = lower_bound(ALL(all_sorted_vols[last_k]), search_vol);
            if(it == all_sorted_vols[last_k].end()) {
                // 1g未満のペイントは無視
            } else {
                int it_ind = distance(all_sorted_vols[last_k].begin(), it);
                temp_rates[last_k] = it_ind;
                double cost = eval_cost(state, temp_rates);
                if(cost < best_cost) {
                    best_cost = cost;
                    best_rates = temp_rates;
                }
            }
        });
    }

    assert((int)best_rates.size() == input.K);
    return best_rates;
}

void solve() {
    const double MAX_TIME = 2800.0;
    TimeKeeper time_keeper(MAX_TIME);

    Input input = parse_input();
    Wall init_wall = struct_init_wall(input);

    State state(init_wall, input);

    ColorMixer mixer(input.own);

    int policy_greedy_cnt = 0;
    for(int i : range(input.H)) {
        // // display
        if(i % 20 == 0) {
            auto [deliver_cost, err_cost, total_cost] = state.get_score();
            cerr << boost::format("Turn: %d, V: %d, E: %d, deliver_cnt: %d, Score: %d (deliver: %d, err: %d)") %
                        state.turn % state.add_cnt % state.error % state.deliver_cnt % total_cost % deliver_cost %
                        err_cost
                 << endl;
        }

        int remain_turn = input.T - state.turn - 30; // 30ターンは余裕を持たせる
        int obj_turn = round((double)remain_turn / (double)(input.H - state.deliver_cnt));

        if(obj_turn >= 9) {
            state.apply_add_less_1gram();
            auto rates = get_rates(input, state, mixer, obj_turn);
            state.apply_color_rates(rates);
        } else {
            PolicyGreedy(state);
            policy_greedy_cnt++;
        }
    }

    if(policy_greedy_cnt > 0) {
        cerr << boost::format("PolicyGreedy %d times.") % policy_greedy_cnt << endl;
    }

    cerr << boost::format("score: %d, elapsed: %f, turn: %d(/%d)]") % get<2>(state.get_score()) %
                time_keeper.getElapsedTime() % state.turn % input.T
         << endl;

    Output output = Output{init_wall, state.actions};
    output.print_output();
}

int main() {
    solve();
    return 0;
}

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
const int INIT_PARTITION_POS = 1; // input_data.N - 1 - 1

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

tuple<Wall, vector<int>> struct_init_wall(Input &input_data) {
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
    return {Wall(wall_h, wall_v), partition_positions};
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
    Wall init_wall;
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
        this->init_wall = init_wall;
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

    void apply_add(const Action &action) {
        this->add_cnt++;
        int id = this->ids[action.i][action.j];
        double w = static_cast<double>(this->paints[id].cap) - this->paints[id].vol;
        if(w <= 1.0) {
            this->paints[id].color = mix(this->paints[id].vol, this->paints[id].color, w, input.own[action.k]);
            this->paints[id].vol = static_cast<double>(this->paints[id].cap);
            cerr << boost::format("Warning: Paint volume exceeds capacity, turn: %d)") % this->turn << endl;
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
            cerr << boost::format("ID: %d, Cap: %d, Vol: %.2f, Color: (%.2f, %.2f, %.2f)") % paint.id % paint.cap % paint.vol % paint.color[0] %
                        paint.color[1] % paint.color[2]
                 << endl;
        }
    }
};

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
    pair<int, int> fractor;
};

struct DicisionAction {
    vector<Action> pre_actions;
    vector<Action> release_actions;
    vector<Action> post_actions;
    vector<int> partition_positions;
    int act_cnt;
    int change_color_num;
};

struct RateItem {
    int k;
    double init_vol;
    vector<pair<int, int>> fractors;
    vector<double> rates;
    vector<double> vols;
    vector<double> post_add_vols;
};

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

    if(state.deliver_cnt <= 990) {
        // 通常は、廃棄=追加コストとみなす
        return err_cost + discard_cost;
    } else {
        // 最後の方は、純粋な追加コストだけみる
        return err_cost + add_cost;
    }
}

DicisionAction dicision_action(Input &input, State &state, ColorMixer &mixer, double obj_turn, vector<int> &partition_positions) {
    const int TOP_N = 10000;
    const int MAX_RESULT = 200;

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

    vector<RateItem> rate_items(input.K);
    for(int k : range(input.K)) {
        Paint paint = state.get_paint(0, k);
        RateItem item;
        int now_partition_pos = partition_positions[k];
        item.k = k;
        item.init_vol = paint.vol;
        item.rates.push_back(1.0);
        item.fractors.push_back(make_pair(1, 1));
        item.vols.push_back(paint.vol);
        item.post_add_vols.push_back(paint.vol + 1.0);
        for(int denominator : range(now_partition_pos + 1, MAX_RATE + 1)) {
            for(int numerator : range(1, denominator)) {
                item.fractors.push_back(make_pair(numerator, denominator));
                double rate = (double)(numerator) / (double)(denominator);
                item.rates.push_back(rate);
                item.vols.push_back(paint.vol * rate);
                item.post_add_vols.push_back((paint.vol + 1.0) * rate);
            }
        }

        vector<int> inds;
        for(int i : range(item.rates.size())) {
            inds.push_back(i);
        }
        sort(ALL(inds), [&](int a, int b) { return item.rates[a] < item.rates[b]; });
        vector<pair<int, int>> sorted_fractors(item.fractors.size());
        vector<double> sorted_rates(item.rates.size());
        vector<double> sorted_vols(item.vols.size());
        vector<double> sorted_post_add_vols(item.post_add_vols.size());
        for(int i : range(item.rates.size())) {
            sorted_fractors[i] = item.fractors[inds[i]];
            sorted_rates[i] = item.rates[inds[i]];
            sorted_vols[i] = item.vols[inds[i]];
            sorted_post_add_vols[i] = item.post_add_vols[inds[i]];
        }

        item.fractors = sorted_fractors;
        item.rates = sorted_rates;
        item.vols = sorted_vols;
        item.post_add_vols = sorted_post_add_vols;
        rate_items[k] = item;
    }

    double best_cost = 1e9;
    vector<ImmediateInfo> best_info;

    for(const auto &result : results) {
        int comb_size = result.indices.size();

        // weightsに近いvolumeを持つペイントのrateを求める(最後のペイントは無視)
        vector<vector<ImmediateInfo>> now_infos(comb_size - 1);

        for(int i : range(comb_size - 1)) {
            int k = result.indices[i];
            double weight = result.weights[i];
            auto &rate_item = rate_items[k];
            int it_ind = -1;
            int rate_item_size = (int)rate_item.rates.size();
            bool is_add = weight > rate_item.init_vol;
            if(!is_add) {
                // addしない場合
                auto it = upper_bound(ALL(rate_item.vols), weight);
                it_ind = distance(rate_item.vols.begin(), it);
            } else {
                // addする場合
                auto it = upper_bound(ALL(rate_item.post_add_vols), weight);
                it_ind = distance(rate_item.post_add_vols.begin(), it);
            }
            if(it_ind >= rate_item_size) {
                it_ind = rate_item_size - 1;
            }

            for(int j : range(-1, 1)) {
                if(it_ind + j < 0 || it_ind + j >= rate_item_size) continue;
                int new_ind = it_ind + j;
                ImmediateInfo info;
                info.k = k;
                info.is_add = is_add;
                info.vol = is_add ? rate_item.post_add_vols[new_ind] : rate_item.vols[new_ind];
                info.fractor = rate_item.fractors[new_ind];
                info.rate = rate_item.rates[new_ind];
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
                auto it = lower_bound(ALL(last_item.vols), search_vol);
                it_ind = distance(last_item.vols.begin(), it);
            } else {
                auto it = lower_bound(ALL(last_item.post_add_vols), search_vol);
                it_ind = distance(last_item.post_add_vols.begin(), it);
            }

            if(it_ind >= (int)last_item.rates.size()) {
                // 1g未満のペイントは無視
            } else {
                ImmediateInfo last_info;
                last_info.k = last_k;
                last_info.is_add = is_add;
                last_info.vol = is_add ? last_item.post_add_vols[it_ind] : last_item.vols[it_ind];
                last_info.fractor = last_item.fractors[it_ind];
                last_info.rate = last_item.rates[it_ind];

                vector<ImmediateInfo> new_info = comb;
                new_info.push_back(last_info);
                double cost = eval_cost(input, state, new_info);
                if(cost < best_cost) {
                    best_cost = cost;
                    best_info = new_info;
                }
            }
        });
    }

    assert((int)best_info.size() != 0);

    DicisionAction action_result;
    action_result.partition_positions = partition_positions;
    action_result.change_color_num = (int)best_info.size();

    // int add_cnt = 0;
    for(auto &info : best_info) {
        int now_partition_pos = partition_positions[info.k];
        if(info.is_add) {
            action_result.pre_actions.push_back(Action::Add(0, info.k, info.k));
        }
        int diff = info.fractor.second - info.fractor.first;
        auto init_act = Action::Toggle(INIT_PARTITION_POS, info.k, INIT_PARTITION_POS + 1, info.k);
        auto release_now_pos_act = Action::Toggle(now_partition_pos, info.k, now_partition_pos + 1, info.k);
        auto denominator_act = Action::Toggle(info.fractor.second - 1, info.k, info.fractor.second, info.k);
        auto numerator_act = Action::Toggle(diff - 1, info.k, diff, info.k);
        if(info.fractor.first == 1 && info.fractor.second == 1) {
            // 全開放
            action_result.release_actions.push_back(release_now_pos_act);
            action_result.post_actions.push_back(init_act);
            action_result.partition_positions[info.k] = INIT_PARTITION_POS;
        } else {
            if(info.fractor.second == now_partition_pos + 1) {
                // 現在の仕切りを動かさない場合
                action_result.pre_actions.push_back(numerator_act);
                action_result.release_actions.push_back(release_now_pos_act);
            } else {
                // 仕切りを広げてから分割するパターン
                action_result.pre_actions.push_back(denominator_act);
                action_result.pre_actions.push_back(release_now_pos_act);
                action_result.pre_actions.push_back(numerator_act);
                action_result.release_actions.push_back(denominator_act);
            }
            if(diff == 1) {
                // 仕切りが小さくなったら復活しておく
                action_result.post_actions.push_back(init_act);
                action_result.post_actions.push_back(numerator_act);
                action_result.partition_positions[info.k] = INIT_PARTITION_POS;
            } else {
                action_result.partition_positions[info.k] = diff - 1;
            }
        }
    }

    int act_cnt = action_result.pre_actions.size() + action_result.release_actions.size() + action_result.post_actions.size();
    action_result.act_cnt = act_cnt;
    return action_result;
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
    const double MAX_TIME = 2800.0;
    TimeKeeper time_keeper(MAX_TIME);

    Input input = parse_input();
    auto [init_wall, partition_positions] = struct_init_wall(input);

    State state(init_wall, input);
    init_state_add_1gram(state, input);

    ColorMixer mixer(input.own);

    // Main Loop
    int policy_greedy_cnt = 0;
    double policy_err_sum = 0.0;
    map<int, int> act_cnt;
    map<int, int> color_cnt;
    for(int h : range(input.H)) {
        if(h % 20 == 0) print_info(state);

        int remain_turn = input.T - state.turn - 30; // 30ターンは余裕を持たせる
        double obj_turn = (double)remain_turn / (double)(input.H - state.deliver_cnt);

        if(obj_turn >= 10.0) {
            auto pre_err = state.error;
            auto action_result = dicision_action(input, state, mixer, obj_turn, partition_positions);

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
            partition_positions = action_result.partition_positions;

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
    output.print_output();
}

int main() {
    solve();
    return 0;
}
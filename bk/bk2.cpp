// ============================================================================
// Template
// ============================================================================
#include <bits/stdc++.h>
using namespace std;

// Judge環境切り替え
#ifndef ONLINE_JUDGE
#define _GLIBCXX_DEBUG
#include <cpp-dump.hpp> // https://zenn.dev/sassan/articles/19db660e4da0a4
#else
#define cpp_dump(...) ;
#endif

// IO高速化
struct IOInit {
    IOInit() {
        ios::sync_with_stdio(0);
        cin.tie(0);
        cout << setprecision(15);
    }
} ioinit;

// 範囲for: [start, end) step
class range {
  public:
    class Iterator {
      public:
        using value_type = int;
        int value, step;

        template <integral T1, integral T2>
        Iterator(T1 value, T2 step) : value(value), step(step) {
        }

        auto operator*() const {
            return value;
        }

        Iterator &operator++() {
            value += step;
            return *this;
        }

        bool operator!=(const Iterator &other) const {
            return step > 0 ? value < other.value : value > other.value;
        }
    };

    template <integral T>
    range(T end) : range(0, end, 1) {
    }
    template <integral T1, integral T2>
    range(T1 start, T2 end) : range(start, end, 1) {
    }
    template <integral T1, integral T2, integral T3>
    range(T1 start, T2 end, T3 step) : begin_(start), end_(end), step_(step) {
        if(step == 0) {
            throw std::invalid_argument("Range step must not be 0");
        }
    }

    Iterator begin() const {
        return Iterator(begin_, step_);
    }
    Iterator end() const {
        return Iterator(end_, step_);
    }

  private:
    int begin_, end_, step_;
};

// Utils
template <typename T>
T intpow(T base, T exp, optional<T> mod = nullopt) {
    T result = 1;
    while(exp > 0) {
        if(exp & 1) {
            if(mod) {
                result = result * base % *mod;
            } else {
                result = result * base;
            }
        }
        exp >>= 1;
        if(exp <= 0) break;
        if(mod) {
            base = base * base % *mod;
        } else {
            base = base * base;
        }
    }
    return result;
}

template <typename T1, typename T2>
inline bool chmin(T1 &a, const T2 &b) {
    bool compare = a > b;
    if(a > b) a = b;
    return compare;
}
template <typename T1, typename T2>
inline bool chmax(T1 &a, const T2 &b) {
    bool compare = a < b;
    if(a < b) a = b;
    return compare;
}

// Set / Multiset
template <typename Set, typename T>
bool erase(Set &s, const T &x) {
    auto itr = s.find(x);
    if(itr != s.end()) {
        s.erase(itr);
        return true;
    }
    return false;
}

// queue / deque (コピーを返すので少しだけ処理が遅いのが不満。)
template <typename Q>
auto pop(Q &q) -> decltype(q.front(), void(), typename Q::value_type{}) {
    auto val = std::move(q.front());
    q.pop_front();
    return val;
}

// priority_queue (同上)
template <typename Q>
auto pop(Q &q) -> decltype(q.top(), void(), typename Q::value_type{}) {
    auto val = std::move(q.top());
    q.pop();
    return val;
}

// ============================================================================
// Huristic Utils
// ============================================================================

#include <atcoder/all>
#include <boost/format.hpp>

using namespace atcoder;

using ll = long long;

#define ALL(obj)  (obj).begin(), (obj).end()
#define RALL(obj) (obj).rbegin(), (obj).rend()

class TimeKeeper {
  private:
    chrono::high_resolution_clock::time_point start_time_;
    double time_threshold_;

  public:
    TimeKeeper(const double &time_threshold)
        : start_time_(chrono::high_resolution_clock::now()), time_threshold_(time_threshold) {
    }

    double getElapsedTime() const {
        auto diff = chrono::high_resolution_clock::now() - this->start_time_;
        return chrono::duration<double, milli>(diff).count();
    }

    bool isTimeOver() const {
        return this->getElapsedTime() >= this->time_threshold_;
    }
};

template <typename Derived, typename UIntType>
class XorshiftBase {
  public:
    using UInt = UIntType;

    UInt next() {
        return static_cast<Derived *>(this)->next();
    }

    // 任意の整数型を返すようテンプレート化（戻り値型を明示）
    UInt randint(UInt max) {
        return next() % max;
    }

    UInt randint(UInt low, UInt high) {
        return low + next() % (high - low + 1);
    }

    double rand() {
        constexpr int bits = std::numeric_limits<UInt>::digits; // 仮数部のbit数ではなく、整数としてのbit数
        constexpr int float_bits = std::numeric_limits<double>::digits; // 仮数部の精度bit数（float=24, double=53）

        if constexpr(bits >= float_bits) {
            UInt value = next() >> (bits - float_bits); // 上位 float_bits を使う
            return static_cast<double>(value) / static_cast<double>(UInt(1) << float_bits);
        } else {
            return static_cast<double>(next()) / static_cast<double>(std::numeric_limits<UInt>::max());
        }
    }

    // 離散分布サンプリング（常に int でOK）
    int sample_discrete(const std::vector<double> &weights) {
        double total = std::accumulate(weights.begin(), weights.end(), 0.0);
        double r = rand() * total;
        double cumulative = 0.0;
        for(size_t i = 0; i < weights.size(); ++i) {
            cumulative += weights[i];
            if(r < cumulative) {
                return static_cast<int>(i);
            }
        }
        return static_cast<int>(weights.size() - 1);
    }

    // イテレータから k 個サンプル（順序ランダム）
    template <typename Iterator>
    std::vector<typename std::iterator_traits<Iterator>::value_type> random_sample(Iterator begin, Iterator end,
                                                                                   int k) {
        using T = typename std::iterator_traits<Iterator>::value_type;
        std::vector<T> pool(begin, end);
        int n = static_cast<int>(pool.size());
        for(int i = 0; i < k; ++i) {
            int j = i + randint(n - i);
            std::swap(pool[i], pool[j]);
        }
        return std::vector<T>(pool.begin(), pool.begin() + k);
    }

    // シャッフル
    template <typename T>
    void shuffle(std::vector<T> &vec) {
        for(int i = (int)(vec.size()) - 1; i > 0; --i) {
            int j = randint(i + 1);
            std::swap(vec[i], vec[j]);
        }
    }
};

class Xorshift32 : public XorshiftBase<Xorshift32, uint32_t> {
  private:
    uint32_t state;

  public:
    explicit Xorshift32(uint32_t seed = 2525) : state(seed) {
    }

    uint32_t next() {
        uint32_t x = state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state = x;
        return x;
    }
};

class Xorshift64 : public XorshiftBase<Xorshift64, uint64_t> {
  private:
    uint64_t state;

  public:
    explicit Xorshift64(uint64_t seed = 202520252025ULL) : state(seed) {
    }

    uint64_t next() {
        uint64_t x = state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        state = x;
        return x;
    }
};

double exponential_schedule(double init, double obj, double elapsed_time, double max_time) {
    double lambda_param = log(obj / init) / max_time;
    return init * exp(lambda_param * elapsed_time);
}

double linear_schedule(double init, double obj, double elapsed_time, double max_time) {
    return init + (obj - init) * (elapsed_time / max_time);
}

// template <typename T>
// tuple<T, double> SimulatedAnnealing(const T &x0, double t0, double t1, double max_time, int display_interval =
// 10000) {
//     T x = x0;
//     T best_x = x;

//     double current_cost = 0; // TODO
//     double best_cost = current_cost;

//     TimeKeeper time_keeper(max_time);

//     int iteration = 0;
//     for(iteration = 0;; iteration++) {
//         int elapsed = time_keeper.getElapsedTime();
//         if(elapsed >= max_time) break;

//         double temp = linear_schedule(t0, t1, elapsed, max_time);

//         T new_x = x;         // TODO
//         double new_cost = 0; // TODO

//         double delta_cost = new_cost - current_cost;
//         if(delta_cost < 0 ||
//            x32rng.rand() < exp(-delta_cost / temp)) { // TODO
//            Tempをexp内に含めると勾配が急になる（≒受理確率が下がる）
//             current_cost = new_cost;
//             x = new_x;
//         }

//         if(current_cost < best_cost) {
//             best_x = x;
//             best_cost = current_cost;
//         }

//         iteration++;
//         if(iteration % display_interval == 0) {
//             cerr << boost::format("Iteration: %d, Current cost: %.2f, Best cost: %.2f, Temp: %.2f") % iteration %
//                         current_cost % best_cost % temp
//                  << endl;
//         }
//     }
//     cerr << boost::format("Iteration: %d, Current cost: %.2f, Best cost: %.2f, Temp: %.2f") % iteration % best_cost
//          << endl;

//     return {best_x, best_cost};
// }

// ============================================================================
// Main
// ============================================================================

using Color = array<double, 3>;

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
    int i, j, k; // For Add
    int i2, j2;  // For Toggle

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
};

struct Input {
    int N, K, H, T, D;
    vector<Color> own;
    vector<Color> target;
};

struct Output {
    vector<vector<bool>> wall_v, wall_h;
    vector<Action> actions;
};

tuple<int, vector<vector<int>>, vector<int>> get_ids(Wall &wall) {
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

Color mix(double v1, Color c1, double v2, Color c2) {
    double sum = v1 + v2;
    if(sum <= 0) return {0.0, 0.0, 0.0};
    return {(v1 * c1[0] + v2 * c2[0]) / sum, (v1 * c1[1] + v2 * c2[1]) / sum, (v1 * c1[2] + v2 * c2[2]) / sum};
}

struct State {
    Input input;
    Wall wall;
    vector<vector<int>> ids;
    vector<int> caps;
    vector<double> vols;
    vector<Color> colors;

    vector<Color> delivered;
    int turn = 0;
    int add_cnt = 0;
    double error = 0.0;
    int deliver_cnt = 0;

    State(const Output &out, const Input &input) {
        this->input = input;
        this->wall = Wall(out.wall_h, out.wall_v);
        auto [ID, ids, caps] = get_ids(wall);
        this->ids = ids;
        this->caps = caps;
        this->vols.assign(ID, 0.0);
        this->colors.assign(ID, {0.0, 0.0, 0.0});
    }

    int get_score() const {
        return 1 + input.D * max(0, this->add_cnt - deliver_cnt) + (int)round(1e4 * this->error);
    }

    void apply(const Action &action) {
        this->turn++;
        if(action.type == ActionType::Add) {
            this->add_cnt++;
            int id = this->ids[action.i][action.j];
            double w = static_cast<double>(this->caps[id]) - this->vols[id];
            if(w <= 1.0) {
                this->colors[id] = mix(this->vols[id], this->colors[id], w, input.own[action.k]);
                this->vols[id] = static_cast<double>(this->caps[id]);
            } else {
                this->colors[id] = mix(this->vols[id], this->colors[id], 1.0, input.own[action.k]);
                this->vols[id] += 1.0;
            }
        } else if(action.type == ActionType::Deliver) {
            this->deliver_cnt++;
            int id = this->ids[action.i][action.j];
            assert((int)this->delivered.size() < input.H);
            assert(this->vols[id] >= 1.0 - 1e-6);
            Color col = this->colors[id];
            Color tgt = input.target[this->delivered.size()];
            this->error += sqrt(pow(col[0] - tgt[0], 2) + pow(col[1] - tgt[1], 2) + pow(col[2] - tgt[2], 2));
            this->vols[id] = max(0.0, this->vols[id] - 1.0);
            this->delivered.push_back(col);
        } else if(action.type == ActionType::Discard) {
            int id = this->ids[action.i][action.j];
            this->vols[id] = max(0.0, this->vols[id] - 1.0);
        } else if(action.type == ActionType::Toggle) {
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
                auto v = this->vols[this->ids[i1][j1]];
                auto vols = vector<double>(ID, 0.0);
                auto colors = vector<Color>(ID, {0.0, 0.0, 0.0});
                for(int i : range(input.N)) {
                    for(int j : range(input.N)) {
                        vols[ids[i][j]] = this->vols[this->ids[i][j]];
                        colors[ids[i][j]] = this->colors[this->ids[i][j]];
                    }
                }
                vols[id1] = v * (double)caps[id1] / (double)(caps[id1] + caps[id2]);
                vols[id2] = v * (double)caps[id2] / (double)(caps[id1] + caps[id2]);
                this->ids = ids;
                this->caps = caps;
                this->vols = vols;
                this->colors = colors;
            } else if(this->ids[i1][j1] != this->ids[i2][j2] && ids[i1][j1] == ids[i2][j2]) {
                auto id = ids[i1][j1];
                auto id1 = this->ids[i1][j1];
                auto id2 = this->ids[i2][j2];
                auto v1 = this->vols[id1];
                auto v2 = this->vols[id2];
                auto c1 = this->colors[id1];
                auto c2 = this->colors[id2];
                auto vols = vector<double>(ID, 0.0);
                auto colors = vector<Color>(ID, {0.0, 0.0, 0.0});
                for(int i : range(input.N)) {
                    for(int j : range(input.N)) {
                        vols[ids[i][j]] = this->vols[this->ids[i][j]];
                        colors[ids[i][j]] = this->colors[this->ids[i][j]];
                    }
                }
                vols[id] = v1 + v2;
                colors[id] = mix(v1, c1, v2, c2);
                this->ids = ids;
                this->caps = caps;
                this->vols = vols;
                this->colors = colors;
            }
        } else {
            cerr << "Unknown action type: " << (int)action.type << endl;
            exit(1);
        }
    }

    void debug() {
        int score = this->get_score();
        cerr << boost::format("Turn: %d, V: %d, E: %d, deliver_cnt: %d, Score: %d") % this->turn % this->add_cnt %
                    this->error % deliver_cnt % score
             << endl;

        // set<int> unique_ids;
        // for(int i : range(input.N)) {
        //     for(int j : range(input.N)) {
        //         auto id = this->ids[i][j];
        //         if(vols[id] < 1e-6) continue; // Skip empty cells
        //         unique_ids.insert(id);
        //     }
        // }

        // for(int id : unique_ids) {
        //     cerr << boost::format("ID: %d, Volume: %.2f, Color: (%.2f, %.2f, %.2f) Capacity %d") % id % vols[id] %
        //                 colors[id][0] % colors[id][1] % colors[id][2] % this->caps[id]
        //          << endl;
        // }
    }
};

int compute_score(const Input &input, const Output &out) {
    State state(out, input);
    for(const auto &a : out.actions) {
        state.debug();
        state.apply(a);
    }
    state.debug();
    return state.get_score();
}

bool validate_output(const Input &input, const Output &out) {
    if((int)out.wall_v.size() != input.N || (int)out.wall_h.size() != input.N - 1) return false;
    for(auto &row : out.wall_v)
        if((int)row.size() != input.N - 1) return false;
    for(auto &row : out.wall_h)
        if((int)row.size() != input.N) return false;

    for(const auto &a : out.actions) {
        if(a.i < 0 || a.i >= input.N || a.j < 0 || a.j >= input.N) return false;
        if(a.type == ActionType::Add && (a.k < 0 || a.k >= input.K)) return false;
        if(a.type == ActionType::Toggle) {
            if(a.i2 < 0 || a.i2 >= input.N || a.j2 < 0 || a.j2 >= input.N) return false;
            int d = abs(a.i - a.i2) + abs(a.j - a.j2);
            if(d != 1) return false;
        }
    }
    return true;
}

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

Output parse_output(const Input &input) {
    int N = input.N;
    Output output;

    output.wall_h = vector<vector<bool>>(N - 1, vector<bool>(N));
    output.wall_v = vector<vector<bool>>(N, vector<bool>(N - 1));

    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N - 1; ++j) {
            int c;
            cin >> c;
            output.wall_v[i][j] = c == 1;
        }
    }
    for(int i = 0; i < N - 1; ++i) {
        for(int j = 0; j < N; ++j) {
            int c;
            cin >> c;
            output.wall_h[i][j] = c == 1;
        }
    }

    int T;
    cin >> T;
    cerr << "T=" << T << endl;
    for(int t = 0; t < T; ++t) {
        int type;
        cin >> type;
        if(type == (int)ActionType::Add) { // add
            int i, j, k;
            cin >> i >> j >> k;
            output.actions.emplace_back(Action::Add(i, j, k));
        } else if(type == (int)ActionType::Deliver) { // deliver
            int i, j;
            cin >> i >> j;
            output.actions.emplace_back(Action::Deliver(i, j));
        } else if(type == (int)ActionType::Discard) { // discard
            int i, j;
            cin >> i >> j;
            output.actions.emplace_back(Action::Discard(i, j));
        } else if(type == (int)ActionType::Toggle) { // toggle
            int i1, j1, i2, j2;
            cin >> i1 >> j1 >> i2 >> j2;
            output.actions.emplace_back(Action::Toggle(i1, j1, i2, j2));
        } else {
            cerr << "Unknown action type int: " << type << endl;
            exit(1);
        }
    }
    return output;
}

int main() {
    Input input = parse_input();
    Output output = parse_output(input);
    if(!validate_output(input, output)) {
        cerr << "Invalid output" << endl;
        return 1;
    }
    cout << compute_score(input, output) << endl;
    return 0;
}

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

// ============================================================================
// Main
// ============================================================================

using Color = array<double, 3>;

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

enum class ActionType { Add, Deliver, Discard, Toggle };
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

vector<vector<int>> get_ids(const vector<vector<bool>> &wall_v, const vector<vector<bool>> &wall_h, vector<int> &caps,
                            int N) {
    vector<vector<int>> ids(N, vector<int>(N, -1));
    int ID = 0;
    caps.clear();

    function<void(int, int)> dfs = [&](int i, int j) {
        ids[i][j] = ID;
        caps.back()++;
        if(j + 1 < N && !wall_v[i][j] && ids[i][j + 1] == -1) dfs(i, j + 1);
        if(i + 1 < N && !wall_h[i][j] && ids[i + 1][j] == -1) dfs(i + 1, j);
        if(j > 0 && !wall_v[i][j - 1] && ids[i][j - 1] == -1) dfs(i, j - 1);
        if(i > 0 && !wall_h[i - 1][j] && ids[i - 1][j] == -1) dfs(i - 1, j);
    };

    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            if(ids[i][j] == -1) {
                caps.push_back(0);
                dfs(i, j);
                ID++;
            }
        }
    }
    return ids;
}

struct State {
    int N;
    vector<vector<bool>> wall_v, wall_h;
    vector<vector<int>> ids;
    vector<int> caps;
    vector<double> vols;
    vector<Color> colors;
    vector<Color> delivered;
    int add_cnt = 0;
    double error = 0.0;

    State(const Output &out, int N) : N(N), wall_v(out.wall_v), wall_h(out.wall_h) {
        ids = get_ids(wall_v, wall_h, caps, N);
        int ID = caps.size();
        vols.assign(ID, 0.0);
        colors.assign(ID, {0.0, 0.0, 0.0});
    }

    static array<double, 3> mix(double v1, array<double, 3> c1, double v2, array<double, 3> c2) {
        double sum = v1 + v2;
        if(sum <= 0) return {0.0, 0.0, 0.0};
        return {(v1 * c1[0] + v2 * c2[0]) / sum, (v1 * c1[1] + v2 * c2[1]) / sum, (v1 * c1[2] + v2 * c2[2]) / sum};
    }

    void debug() const {
        cerr << "== State Debug ==\n";
        for(int id = 0; id < (int)vols.size(); ++id) {
            cerr << "Well " << id << ": vol=" << vols[id] << ", color=(" << colors[id][0] << ", " << colors[id][1]
                 << ", " << colors[id][2] << ")\n";
        }
    }

    bool apply(const Input &input, const Action &action) {
        if(action.type == ActionType::Add) {
            add_cnt++;
            int id = ids[action.i][action.j];
            double space = caps[id] - vols[id];
            double add = min(1.0, space);
            colors[id] = mix(vols[id], colors[id], add, input.own[action.k]);
            vols[id] += add;
        } else if(action.type == ActionType::Deliver) {
            int id = ids[action.i][action.j];
            if(vols[id] < 1.0 - 1e-6) return false;
            array<double, 3> col = colors[id];
            array<double, 3> tgt = input.target[delivered.size()];
            error += sqrt(pow(col[0] - tgt[0], 2) + pow(col[1] - tgt[1], 2) + pow(col[2] - tgt[2], 2));
            vols[id] = max(0.0, vols[id] - 1.0);
            delivered.push_back(col);
        } else if(action.type == ActionType::Discard) {
            int id = ids[action.i][action.j];
            vols[id] = max(0.0, vols[id] - 1.0);
        } else if(action.type == ActionType::Toggle) {
            int i1 = action.i, j1 = action.j;
            int i2 = action.i2, j2 = action.j2;
            if(i1 == i2)
                wall_v[i1][min(j1, j2)] = wall_v[i1][min(j1, j2)] != true;
            else
                wall_h[min(i1, i2)][j1] = wall_h[min(i1, i2)][j1] != true;
            ids = get_ids(wall_v, wall_h, caps, N);
        }
        // debug();  // Uncomment for step-by-step debug output
        return true;
    }
};

int compute_score(const Input &input, const Output &out) {
    State state(out, input.N);
    for(const auto &a : out.actions) {
        if(!state.apply(input, a)) return 0;
    }
    if((int)state.delivered.size() < input.H) return 0;
    return 1 + input.D * (state.add_cnt - input.H) + (int)round(1e4 * state.error);
}

class Wall {
  public:
    vector<vector<bool>> wall_h;
    vector<vector<bool>> wall_v;
    int N;

    Wall(vector<vector<bool>> &horizontal, vector<vector<bool>> &vertical) {
        // check size
        int horizontal_h = horizontal.size();
        int horizontal_w = horizontal[0].size();
        int vertical_h = vertical.size();
        int vertical_w = vertical[0].size();
        assert(horizontal_h == horizontal_w + 1);
        assert(vertical_h + 1 == vertical_w);
        assert(horizontal_h == vertical_w);

        this->wall_h = horizontal;
        this->wall_v = vertical;
        this->N = horizontal_h;
    }

    void switch_horizontal(int i, int j) {
        wall_h[i][j] = !wall_h[i][j];
    }

    void switch_vertical(int i, int j) {
        wall_v[i][j] = !wall_v[i][j];
    }
};

Wall struct_init_wall(Input &input_data) {
    vector<vector<bool>> horizontal(input_data.N, vector<bool>(input_data.N - 1, false));
    vector<vector<bool>> vertical(input_data.N - 1, vector<bool>(input_data.N, false));

    for(int x : range(input_data.N - 1)) {
        for(int y : range(input_data.N - 1)) {
            horizontal[y][x] = true;
        }
    }
    for(int x : range(input_data.N)) {
        vertical[input_data.N - 1 - 1][x] = true;
    }

    return Wall(horizontal, vertical);
}

void output_partition(const Wall &partition) {
    for(int i : range(partition.N)) {
        for(int j : range(partition.N - 1)) {
            cout << (partition.wall_h[i][j] ? "1" : "0") << " ";
        }
        cout << endl;
    }
    for(int i : range(partition.N - 1)) {
        for(int j : range(partition.N)) {
            cout << (partition.wall_v[i][j] ? "1" : "0") << " ";
        }
        cout << endl;
    }
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
    Output output;
    output.wall_v = vector<vector<bool>>(input.N, vector<bool>(input.N - 1));
    output.wall_h = vector<vector<bool>>(input.N - 1, vector<bool>(input.N));
    for(int i = 0; i < input.N; ++i) {
        for(int j = 0; j < input.N - 1; ++j) {
            int v;
            cin >> v;
            output.wall_v[i][j] = v;
        }
    }
    for(int i = 0; i < input.N - 1; ++i) {
        for(int j = 0; j < input.N; ++j) {
            int v;
            cin >> v;
            output.wall_h[i][j] = v;
        }
    }
    int num_actions;
    cin >> num_actions;
    for(int i = 0; i < num_actions; ++i) {
        string s;
        cin >> s;
        if(s == "add") {
            int x, y, k;
            cin >> x >> y >> k;
            output.actions.push_back(Action::Add(x, y, k));
        } else if(s == "deliver") {
            int x, y;
            cin >> x >> y;
            output.actions.push_back(Action::Deliver(x, y));
        } else if(s == "discard") {
            int x, y;
            cin >> x >> y;
            output.actions.push_back(Action::Discard(x, y));
        } else if(s == "toggle") {
            int x1, y1, x2, y2;
            cin >> x1 >> y1 >> x2 >> y2;
            output.actions.push_back(Action::Toggle(x1, y1, x2, y2));
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

// class Env {
//   public:
//     InputData input_data;
//     Partition partition;

//     int turn = 0;
//     int now_target_color = 0;
//     int add_color_cnt = 0;

//     double color_error = 0.0;
//     double add_color_cost = 0.0;

//     Env(InputData &input_data, Partition &partition) : input_data(input_data), partition(partition) {
//     }

//     void print_info() {
//         cerr << boost::format("C: %d, T: %d, Add: %d, Err: %f, Err(Color): %f, Err(Add): %f") % now_target_color %
//                     turn % add_color_cnt % (color_error + add_color_cost) % color_error % add_color_cost
//              << endl;
//     }

//     void op1(int i, int j, int k) {
//         // 絵の具追加
//         turn++;
//         add_color_cnt++;

//         cerr << boost::format("[op1] %d %d %d") % i % j % k << endl;
//         print_info();

//         cout << boost::format("1 %d %d %d") % i % j % k << endl;
//     }

//     void op2(int i, int j) {
//         // 絵の具を渡す
//         turn++;
//         now_target_color++;

//         cerr << boost::format("[op2] %d %d") % i % j << endl;
//         print_info();

//         cout << boost::format("2 %d %d") % i % j << endl;
//     }

//     void op3(int i, int j) {
//         // 絵の具追加

//         turn++;

//         cerr << boost::format("[op3] %d %d") % i % j << endl;
//         print_info();

//         cout << boost::format("3 %d %d") % i % j << endl;
//     }

//     void op4(int i1, int i2, int j1, int j2) {
//         // 仕切り変更
//         if(i1 == i2) {
//             partition.switch_horizontal(i1, j1);
//         } else if(j1 == j2) {
//             partition.switch_vertical(i1, j1);
//         } else {
//             cerr << "Invalid operation: op4 with different i and j indices." << endl;
//             exit(1);
//         }

//         cout << boost::format("4 %d %d %d %d") % i1 % i2 % j1 % j2 << endl;
//         cerr << boost::format("[op4] %d %d %d %d") % i1 % i2 % j1 % j2 << endl;
//         print_info();

//         turn++;
//     }
// };

// InputData read_input() {
//     InputData data;
//     cin >> data.N >> data.K >> data.H >> data.T >> data.D;

//     data.own.resize(data.K);
//     for(auto &c : data.own) {
//         for(auto &v : c)
//             cin >> v;
//     }

//     data.target.resize(data.H);
//     for(auto &c : data.target) {
//         for(auto &v : c)
//             cin >> v;
//     }

//     // Debug output
//     cerr << "===" << endl;
//     cerr << boost::format("N: %d, K: %d, H: %d, T: %d, D: %d") % data.N % data.K % data.H % data.T % data.D << endl;
//     cerr << "Own Colors: " << endl;
//     for(const auto &c : data.own) {
//         cerr << boost::format("(%f, %f, %f)") % c[0] % c[1] % c[2] << endl;
//     }

//     cerr << "Target Colors: " << endl;
//     for(int i : range(3)) {
//         auto c = data.target[i];
//         cerr << boost::format("(%f, %f, %f)") % c[0] % c[1] % c[2] << endl;
//     }
//     cerr << "..." << endl;
//     cerr << "===" << endl;

//     return data;
// }

// void solve() {
//     InputData input_data = read_input();
//     Partition partition = init_partition(input_data);

//     output_partition(partition);
//     Env env(input_data, partition);

//     for(int h : range(input_data.H)) {
//         env.op1(0, 0, 0);
//         env.op2(0, 0);
//     }

//     cerr << "===" << endl;
//     env.print_info();

//     // やること
//     // 1g未満のグループに絵の具を追加
//     // 良い比率を求める（annealingで）
//     //
// }

// int main() {
//     solve();
// }
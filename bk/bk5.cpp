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

Xorshift32 x32rng;
Xorshift64 x64rng;

// ============================================================================
// Main
// ============================================================================

const int MAX_RATE = 19;
using Color = array<double, 3>;

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
        // while(get_paint(input.N - 1, 0).vol > 1e-6) {
        //     this->apply(Action::Discard(input.N - 1, 0));
        // }

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

vector<int> init_x(State &state) {
    vector<int> rates(state.input.K, 0);
    for(int k : range(state.input.K)) {
        rates[k] = x64rng.randint(0, MAX_RATE);
    }
    return rates;
}

double eval_cost(State &state, const vector<int> &rates, bool debug = false) {
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
    // double discard_cost = max(0.0, sum_vol - 1.0) * (double)(state.input.D);
    double discard_cost = 0.0;
    double less_1gram_cost = 0.0;
    if(sum_vol < 1.0) {
        less_1gram_cost = (1.0 - sum_vol) * 1e7 + 1e8; // 1g未満のペイントは高コスト
    }
    if(debug) {
        cerr << boost::format(
                    "Mixed Color: (%.2f, %.2f, %.2f), Target: (%.2f, %.2f, %.2f), "
                    "Error Cost: %.2f, Discard Cost: %.2f, Less 1g Cost: %.2f") %
                    mixed_color[0] % mixed_color[1] % mixed_color[2] % now_target[0] % now_target[1] % now_target[2] %
                    err_cost % discard_cost % less_1gram_cost
             << endl;
    }

    return err_cost + discard_cost + less_1gram_cost;
}

vector<int> neighbor_rates(const vector<int> &rates) {
    vector<int> new_rates = rates;
    const int n = (int)rates.size();
    if(x64rng.rand() < 0.5) {
        int k = x64rng.randint(0, n - 1);
        if(new_rates[k] == 0) {
            new_rates[k]++;
        } else if(new_rates[k] == MAX_RATE) {
            new_rates[k]--;
        } else {
            if(x64rng.rand() < 0.5) {
                new_rates[k]++;
            } else {
                new_rates[k]--;
            }
        }
    } else {
        int k1 = x64rng.randint(0, n - 1);
        int k2 = x64rng.randint(0, n - 1);
        if(k1 != k2) {
            swap(new_rates[k1], new_rates[k2]);
        }
    }
    return new_rates;
};

tuple<vector<int>, double> PolicyAnnealing(State &state, const vector<int> &x0, double t0, double t1, double max_time,
                                           bool is_display, int display_interval = 100000) {
    vector<int> x = x0;
    vector<int> best_x = x;

    double current_cost = eval_cost(state, x);
    double best_cost = current_cost;

    TimeKeeper time_keeper(max_time);

    int iteration = 0;
    for(iteration = 0;; iteration++) {
        double elapsed = time_keeper.getElapsedTime();
        if(elapsed >= max_time) break;

        double temp = linear_schedule(t0, t1, elapsed, max_time);

        vector<int> new_x = neighbor_rates(x);
        double new_cost = eval_cost(state, new_x);

        double delta_cost = new_cost - current_cost;

        // !NOTE: Tempをexp内に含めると勾配が急になる（≒受理確率が下がる） current_cost = new_cost;
        if(delta_cost < 0 || x32rng.rand() < exp(-delta_cost / temp)) {
            x = new_x;
            current_cost = new_cost;
        }

        if(current_cost < best_cost) {
            best_x = x;
            best_cost = current_cost;
        }

        if(is_display && iteration % display_interval == 0) {
            cerr << boost::format("Iteration: %d, Current cost: %.2f, Best cost: %.2f, Temp: %.2f") % iteration %
                        current_cost % best_cost % temp
                 << endl;
        }
    }

    if(is_display) {
        cerr << boost::format("Iteration: %d, Best cost: %.2f") % iteration % best_cost << endl;
    }

    if(best_cost > 1e8) {
        cerr << "Warning: Best cost is too high, may not be a valid solution." << endl;
        cpp_dump(best_x, best_cost, current_cost, x);
        for(int k : range(state.input.K)) {
            cpp_dump(state.get_paint(0, k).color, state.get_paint(0, k).vol);
        }
        eval_cost(state, best_x, true);
        eval_cost(state, x, true);
        exit(1);
    }

    return {best_x, best_cost};
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

void solve() {
    const double MAX_TIME = 2800.0;

    Input input = parse_input();
    Wall init_wall = struct_init_wall(input);

    State state(init_wall, input);

    TimeKeeper time_keeper(MAX_TIME);
    double now_calc_time = 2.75;
    for(int i : range(input.H)) {
        if(i % 20 == 0) {
            auto [deliver_cost, err_cost, total_cost] = state.get_score();
            cerr << boost::format("Turn: %d, V: %d, E: %d, deliver_cnt: %d, Score: %d (deliver: %d, err: %d)") %
                        state.turn % state.add_cnt % state.error % state.deliver_cnt % total_cost % deliver_cost %
                        err_cost
                 << endl;
        }

        int remain_turn = state.input.T - state.turn - 100; // 100ターンは余裕を持たせる
        int obj_turn = round((double)remain_turn / (double)(state.input.H - state.deliver_cnt));

        if(obj_turn >= 10) {
            double elapsed = time_keeper.getElapsedTime();
            int remain_h = input.H - i;
            double avg_time = (MAX_TIME - elapsed) / remain_h;
            if(avg_time < now_calc_time) {
                now_calc_time = max(2.50, now_calc_time - 0.05);
            } else {
                now_calc_time = min(3.00, now_calc_time + 0.05);
            }

            state.apply_add_less_1gram();
            auto x = init_x(state);
            auto [rates, cost] = PolicyAnnealing(state, x, 100.0, 1e-6, now_calc_time, false, 100000);
            state.apply_color_rates(rates);
        } else {
            PolicyGreedy(state);
        }
    }

    cerr << boost::format("Final Score: %d [elapsed: %f]") % get<2>(state.get_score()) % time_keeper.getElapsedTime()
         << endl;

    Output output = Output{init_wall, state.actions};
    output.print_output();
}

int main() {
    solve();
    return 0;
}


// =========================================================
// Common
// =========================================================
#include <bits/stdc++.h>
using namespace std;

#include <boost/format.hpp>

// Judge環境切り替え
#ifndef ONLINE_JUDGE
#define _GLIBCXX_DEBUG
#include <cpp-dump.hpp>
#else
#define cpp_dump(...) ;
#endif

using ll = long long;
using Color = array<double, 3>;
using Fractor = pair<int, int>;
using Fractors = vector<Fractor>;

#define ALL(obj)  (obj).begin(), (obj).end()
#define RALL(obj) (obj).rbegin(), (obj).rend()
// Skipped: common.hpp already included
// Skipped: common.hpp already included

// =========================================================
// Utils
// =========================================================

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

// ハッシュ（https://qiita.com/hamamu/items/4d081751b69aa3bb3557）
template <class T>
size_t HashCombine(const size_t seed, const T &v) {
    return seed ^ (std::hash<T>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}
template <class T, class S>
struct std::hash<std::pair<T, S>> {
    size_t operator()(const std::pair<T, S> &keyval) const noexcept {
        return HashCombine(std::hash<T>()(keyval.first), keyval.second);
    }
};
template <class T>
struct std::hash<std::vector<T>> {
    size_t operator()(const std::vector<T> &keyval) const noexcept {
        size_t s = 0;
        for(auto &&v : keyval)
            s = HashCombine(s, v);
        return s;
    }
};
template <int N>
struct HashTupleCore {
    template <class Tuple>
    size_t operator()(const Tuple &keyval) const noexcept {
        size_t s = HashTupleCore<N - 1>()(keyval);
        return HashCombine(s, std::get<N - 1>(keyval));
    }
};
template <>
struct HashTupleCore<0> {
    template <class Tuple>
    size_t operator()(const Tuple &keyval) const noexcept {
        return 0;
    }
};
template <class... Args>
struct std::hash<std::tuple<Args...>> {
    size_t operator()(const tuple<Args...> &keyval) const noexcept {
        return HashTupleCore<tuple_size<tuple<Args...>>::value>()(keyval);
    }
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

class TimeKeeper {
  private:
    // high_resolution_clock → steady_clock に変更
    std::chrono::steady_clock::time_point start_time_;
    double time_threshold_;

  public:
    TimeKeeper(double time_threshold) : start_time_(std::chrono::steady_clock::now()), time_threshold_(time_threshold) {
    }

    double getElapsedTime() const {
        auto diff = std::chrono::steady_clock::now() - start_time_;
        return std::chrono::duration<double, std::milli>(diff).count();
    }

    bool isTimeOver() const {
        return getElapsedTime() >= time_threshold_;
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
        constexpr int bits = std::numeric_limits<UInt>::digits;         // 仮数部のbit数ではなく、整数としてのbit数
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
    std::vector<typename std::iterator_traits<Iterator>::value_type> random_sample(Iterator begin, Iterator end, int k) {
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

// 直積を生成する
template <typename T, typename Func>
void cartesian_product(const std::vector<std::vector<T>> &vectors, Func callback) {
    int n = vectors.size();
    std::vector<int> indices(n, 0);
    std::vector<T> result(n);

    while(true) {
        for(int i = 0; i < n; ++i) {
            result[i] = vectors[i][indices[i]];
        }
        callback(result); // ラムダが自動的に推論される

        int k = n - 1;
        while(k >= 0) {
            indices[k]++;
            if(indices[k] < static_cast<int>(vectors[k].size())) break;
            indices[k] = 0;
            --k;
        }
        if(k < 0) break;
    }
}

double exponential_schedule(double init, double obj, double elapsed_time, double max_time) {
    double lambda_param = log(obj / init) / max_time;
    return init * exp(lambda_param * elapsed_time);
}

double linear_schedule(double init, double obj, double elapsed_time, double max_time) {
    return init + (obj - init) * (elapsed_time / max_time);
}

pair<int, int> reduce_fraction(pair<int, int> frac) {
    int num = frac.first;
    int den = frac.second;

    if(den == 0) throw invalid_argument("Denominator cannot be zero");

    int g = gcd(abs(num), abs(den));
    num /= g;
    den /= g;

    return {num, den};
}

pair<int, int> mul_fracs(vector<pair<int, int>> fracs) {
    int num = 1;
    int den = 1;
    for(const auto &frac : fracs) {
        num *= frac.first;
        den *= frac.second;
    }
    return reduce_fraction({num, den});
}

template <typename RefT>
std::vector<size_t> make_sorted_indices(const std::vector<RefT> &ref, bool descending = false) {
    std::vector<size_t> indices(ref.size());
    for(size_t i = 0; i < ref.size(); ++i)
        indices[i] = i;

    std::sort(indices.begin(), indices.end(), [&](size_t i, size_t j) { return descending ? ref[i] > ref[j] : ref[i] < ref[j]; });

    return indices;
}

template <typename T>
void reorder_vector(std::vector<T> &vec, const std::vector<size_t> &indices) {
    std::vector<T> reordered(vec.size());
    for(size_t i = 0; i < indices.size(); ++i) {
        reordered[i] = vec[indices[i]];
    }
    vec = std::move(reordered);
}

void choose_front(int start, int needed, int m, std::vector<int> &sel, std::vector<std::vector<int>> &result_list) {
    if(needed == 0) {
        std::vector<int> full = sel;
        full.push_back(m);
        result_list.push_back(full);
        return;
    }

    for(int i = start; i <= m - needed; ++i) {
        sel.push_back(i);
        choose_front(i + 1, needed - 1, m, sel, result_list);
        sel.pop_back();
    }
}

vector<vector<int>> choose_nCk(const int N, const int K, int max_comb = 10000) {
    std::vector<int> buffer;
    std::vector<std::vector<int>> tmp;
    vector<vector<int>> comb_list;
    for(int m = K - 1; m < N; ++m) {
        tmp.clear();
        buffer.clear();
        choose_front(0, K - 1, m, buffer, tmp);
        for(auto &comb : tmp) {
            if((int)comb_list.size() >= max_comb) {
                return comb_list;
            }
            comb_list.push_back(comb);
        }
    }

    return comb_list;
}

Xorshift64 xor_rng;
// =========================================================
// Game
// =========================================================

struct Input {
    int N, K, H, T, D;
    vector<Color> own;
    vector<Color> target;
};

double eval_error(Color col, Color tgt) {
    return sqrt(pow(col[0] - tgt[0], 2) + pow(col[1] - tgt[1], 2) + pow(col[2] - tgt[2], 2));
}

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
            throw runtime_error("Unknown ActionType!");
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
            throw runtime_error("Unknown ActionType!");
        }
    }
};

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

            caps.emplace_back(cap);
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
    double discard = 0.0;
    int deliver_cnt = 0;
    int discard_cnt = 0;

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
        if(w < 1.0) {
            this->paints[id].color = mix(this->paints[id].vol, this->paints[id].color, w, input.own[action.k]);
            this->paints[id].vol = static_cast<double>(this->paints[id].cap);
            throw runtime_error(boost::str(boost::format("Error: Paint volume exceeds capacity, turn: %d)") % this->turn));
        } else {
            this->paints[id].color = mix(this->paints[id].vol, this->paints[id].color, 1.0, input.own[action.k]);
            this->paints[id].vol += 1.0;
        }
    }

    void apply_deliver(const Action &action) {
        this->deliver_cnt++;
        int id = this->ids[action.i][action.j];
        if((int)this->delivered.size() >= input.H) {
            throw runtime_error("Error: Too many deliveries.");
        };
        if(this->paints[id].vol < 1.0 - 1e-6) {
            throw runtime_error("Error: Not enough paint to deliver.");
        };
        Color col = this->paints[id].color;
        Color tgt = input.target[this->delivered.size()];
        this->error += eval_error(col, tgt);
        this->paints[id].vol = max(0.0, this->paints[id].vol - 1.0);
        this->delivered.emplace_back(col);
    }

    void apply_discard(const Action &action) {
        this->discard_cnt++;
        int id = this->ids[action.i][action.j];
        if(this->paints[id].vol < 1e-6) {
            throw runtime_error("Error: Not enough paint to discard.");
        };
        discard += min(1.0, this->paints[id].vol);
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
            throw runtime_error("Error: Too many turns.");
        }

        this->turn++;
        this->actions.emplace_back(action);

        if(action.type == ActionType::Add) {
            this->apply_add(action);
        } else if(action.type == ActionType::Deliver) {
            this->apply_deliver(action);
        } else if(action.type == ActionType::Discard) {
            this->apply_discard(action);
        } else if(action.type == ActionType::Toggle) {
            this->apply_toggle(action);
        } else {
            throw runtime_error("Unknown action type.");
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
};// Skipped: common.hpp already included
// Skipped: game.hpp already included

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

// Skipped: common.hpp already included
// Skipped: utils.hpp already included

// ====================================
// NNLSを解くためのクラス
// ====================================

#include <Eigen/Core>
#include <Eigen/Dense>

// 単純体への射影関数
Eigen::VectorXd ProjectOntoSimplex(const Eigen::VectorXd& v) {
    const int n = v.size();
    std::vector<double> u(n);
    for(int i = 0; i < n; ++i)
        u[i] = v[i];
    std::sort(u.begin(), u.end(), std::greater<double>());

    std::vector<double> cumsum(n);
    cumsum[0] = u[0];
    for(int i = 1; i < n; ++i)
        cumsum[i] = cumsum[i - 1] + u[i];

    int rho = -1;
    double theta = 0;
    for(int j = 0; j < n; ++j) {
        double t = (cumsum[j] - 1.0) / (j + 1);
        if(u[j] - t > 0) {
            rho = j;
            theta = t;
        }
    }
    if(rho < 0) {
        return Eigen::VectorXd::Constant(n, 1.0 / n);
    }
    Eigen::VectorXd w(n);
    for(int i = 0; i < n; ++i) {
        w[i] = std::max(v[i] - theta, 0.0);
    }
    return w;
}

// -----------------------------------------------------------------------------
// estimateMaxEigenvalue()
//   パワーイテレーションにより、ATA = Aᵀ A の最大固有値を推定する。
//   A: (m×n) 行列、powerIter: イテレーション回数（10～20程度で十分）
// -----------------------------------------------------------------------------
double estimateMaxEigenvalue(const Eigen::MatrixXd& A, int powerIter = 20) {
    const int n = A.cols();
    // Aᵀ A に対するパワー法
    Eigen::VectorXd v = Eigen::VectorXd::Random(n);
    v.normalize();
    for(int it = 0; it < powerIter; ++it) {
        // w ← (Aᵀ A) v
        Eigen::VectorXd w = A.transpose() * (A * v);
        double wnorm = w.norm();
        if(wnorm <= 0) break;
        v = w / wnorm;
    }
    // λ ≈ vᵀ (Aᵀ A) v
    Eigen::VectorXd Av = A * v;
    Eigen::VectorXd ATAv = A.transpose() * Av;
    double lambda = v.dot(ATAv);
    return lambda;
}

// -----------------------------------------------------------------------------
// nnls_projected_gradient_bb_clipped()
//   A ∈ R^{m×n}, b ∈ R^m を与えて、
//   min_{x ∈ simplex} ½ ||A x − b||^2 をクリップ付き BB ステップ幅で解く。
//   ・x は「x_i >=0, sum_i x_i = 1」を常に満たす（ProjectOntoSimplexを挟む）。
//   tol: KKT 条件残差許容値
//   max_iter: イテレーション上限
// -----------------------------------------------------------------------------
bool nnls_projected_gradient_bb_clipped(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                                        Eigen::VectorXd& x, // 初期 guess を与え、解がここに返る (size n)
                                        double tol = 1e-7,  // KKT 条件の残差閾値
                                        int max_iter = 1e3, // 最大イテレーション数
                                        bool is_alpha_max_fixed = true) {
    const int m = A.rows();
    const int n = A.cols();
    if(b.size() != m || x.size() != n) {
        cerr << "[nnls_pg_bb] サイズ不一致: A(" << m << "×" << n << "), b(" << b.size() << "), x(" << x.size() << ")\n";
        return false;
    }

    // 1) 事前計算: ATA, ATb
    Eigen::MatrixXd ATA = A.transpose() * A; // (n×n)
    Eigen::VectorXd ATb = A.transpose() * b; // (n)

    // 2) λ_max = 最大固有値(AᵀA) を推定（パワー法）
    double lambda_max = estimateMaxEigenvalue(A, 20);
    if(lambda_max <= 0) lambda_max = 1e-3; // 念のためゼロ割回避

    // 3) α_max, α_min の設定（上限・下限をクリップする）
    //    上限: 0.8 / λ_max  (「1/λ_max の約80%」)
    //    下限: 1e-6 / λ_max
    double alpha_max;
    if(is_alpha_max_fixed) {
        alpha_max = 1e8; // alphaがでかいと発散することがあるので注意
    } else {
        alpha_max = 0.99 / lambda_max;
    }
    const double alpha_min = 1e-12 / lambda_max;

    // 4) 初期化: x >= 0 且つ sum(x)=1 にする
    //    └  もし呼び出し側が x ≥ 0, sum=1 を用意していなければ、
    //        ここで一様分布に初期化してもよい。
    x = ProjectOntoSimplex(x);

    // 5) 初期勾配 g = ATA*x - ATb
    Eigen::VectorXd g = ATA * x - ATb;

    // 6) 初期 step size: 1 / λ_max
    double alpha = 1.0 / lambda_max;

    // (反復用テンポラリ)
    Eigen::VectorXd x_prev(n), g_prev(n), s(n), y_vec(n);

    for(int iter = 0; iter < max_iter; ++iter) {
        // (a) 前回の保存
        x_prev = x;
        g_prev = g;

        // (b) 勾配ステップ
        Eigen::VectorXd x_tent = x - alpha * g;

        // (c) 単純体への射影 (sum=1, x_i>=0 を維持)
        x = ProjectOntoSimplex(x_tent);

        // (d) 勾配の再計算
        g = ATA * x - ATb;

        // (e) KKT 条件による収束判定
        //       「最適性残差 = max_i |min(x_i, g_i)|」が tol 未満なら終了
        double kkt_res = 0.0;
        for(int i = 0; i < n; ++i) {
            double xi = x[i], gi = g[i];
            // x_i > 0 なら g_i ≈ 0、x_i = 0 なら g_i ≥ 0
            double tmp = std::min(xi, gi);
            kkt_res = std::max(kkt_res, std::abs(tmp));
        }
        if(kkt_res < tol) {
            // 収束した
            // cout << "[BB-PGD] iter=" << iter << "  KKT_res=" << kkt_res << "\n";
            return true;
        }

        // (f) BBステップ幅更新
        s = x - x_prev;     // Δx
        y_vec = g - g_prev; // Δg
        double sty = s.dot(y_vec);
        double sts = s.squaredNorm();
        double alpha_bb;
        if(sty > 1e-16) {
            alpha_bb = sts / sty;
        } else {
            // 分母が非常に小さい・負になるときは小さな α_min を使っておく
            alpha_bb = alpha_min;
        }
        // (g) α をクリップ
        alpha = std::min(std::max(alpha_bb, alpha_min), alpha_max);
    }

    // max_iter に到達しても収束せず
    return false;
}

vector<vector<int>> construct_subsets(int size, int k) {
    vector<vector<int>> subsets;
    vector<int> comb(size);
    function<void(int, int)> dfs = [&](int start, int depth) {
        if(depth == size) {
            subsets.emplace_back(comb.begin(), comb.end());
            return;
        }
        for(int x = start; x < k; x++) {
            comb[depth] = x;
            dfs(x + 1, depth + 1);
        }
    };
    dfs(0, 0);

    return subsets;
}

vector<vector<double>> Gram;
vector<vector<double>> pseudo;
vector<vector<double>> invG;

class ColorMixer {
  public:
    struct Result {
        double err;
        vector<int> indices;
        vector<double> weights;

        bool operator<(Result const& o) const {
            return err < o.err;
        }
    };

    struct SubsetInfo {
        int size;
        vector<int> indices;
        vector<vector<double>> Gram;   // Gram 行列: size×size
        vector<vector<double>> pseudo; // 擬似逆行列: size×3
    };

    static constexpr double EPS = 1e-7;         // 許容誤差 (sum_w ≈ 1.0 ± epsに収束)
    static constexpr int MAX_ITER = 50;         // 簡易評価
    static constexpr int MAX_ITER_HEAVY = 1000; // 最大反復回数

    vector<Color> paints;
    int K;

    ColorMixer(const vector<Color>& paints_input) : paints(paints_input) {
        K = paints.size();
    }

    vector<Result> solve_nnls(const Color& t, int comb_size, int find_top_n) {
        auto subsets = construct_subsets(comb_size, this->K);
        sort(ALL(subsets), [&](auto& a, auto& b) { return xor_rng.next() < 0.5; });

        const int MAX_SUBSETS = 2000;
        const int MAX_HEAVY_NNLS = 1;
        if(subsets.size() > MAX_SUBSETS) {
            subsets.resize(MAX_SUBSETS);
        }

        // const int TEMP_HEAP_SIZE = 30;

        priority_queue<Result> heap;
        for(const auto& indices : subsets) {
            Result r = solve_nnls_inv(t, indices);
            if((int)heap.size() < find_top_n) {
                heap.push(r);
            } else if(r.err < heap.top().err) {
                heap.pop();
                heap.push(r);
            }
        }

        vector<Result> results;
        while(!heap.empty()) {
            auto r = heap.top();
            heap.pop();
            results.push_back(r);
            Result r2 = solve_nnls_pdm(r.indices, t, true, EPS, MAX_ITER);
            if(r2.err < r.err) {
                results.push_back(r2);
            } else {
                results.push_back(r);
            }
        }
        sort(results.begin(), results.end(), [](const Result& a, const Result& b) { return a.err < b.err; });

        for(int i : range(min((int)results.size(), MAX_HEAVY_NNLS))) {
            auto& r = results[i];
            if(r.err < 1e-4) continue;
            Result r3 = solve_nnls_pdm(r.indices, t, false, EPS, MAX_ITER_HEAVY);
            if(r3.err < r.err) {
                results[i] = r3;
            }
        }
        return results;
    }

    Result solve_nnls_pdm(const vector<int>& indices, const Color& t_color, double is_alpha_max_fixed = true, double eps = EPS, int max_iter = MAX_ITER) {
        int n = static_cast<int>(indices.size());
        Eigen::Vector3d t(t_color[0], t_color[1], t_color[2]);
        Eigen::VectorXd x0 = Eigen::VectorXd::Constant(n, 1.0 / double(n));
        Eigen::MatrixXd A;

        A.resize(3, n);
        for(int j = 0; j < n; ++j) {
            int paint_idx = indices[j];
            const auto& col = paints[paint_idx];
            Eigen::Vector3d c(col[0], col[1], col[2]);
            A.col(j) = c;
        }

        Eigen::VectorXd w;

        bool ok = nnls_projected_gradient_bb_clipped(A, t, x0, eps, max_iter, is_alpha_max_fixed);
        auto err = (A * x0 - t).norm();
        w = x0.transpose();

        double sum_w = w.sum();
        assert(abs(sum_w - 1.0) < 1e-6); // 合計が 1 に正規化されていることを確認

        vector<double> weights;
        for(int i = 0; i < n; ++i) {
            weights.push_back(w(i));
        };

        return Result{err, indices, weights};
    }

    Result solve_nnls_inv(const Color& t, vector<int> indices) {
        int n = static_cast<int>(indices.size());

        calc_gram_inv(indices);

        double t_norm2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];

        // 1) 擬似逆行列 × t で制約なし最小二乗解を得る
        vector<double> w_ls(n, 0.0);
        for(int i = 0; i < n; i++) {
            // pseudo はサイズ n×3 の行列
            w_ls[i] = pseudo[i][0] * t[0] + pseudo[i][1] * t[1] + pseudo[i][2] * t[2];
        }

        // 2) クリッピング＆正規化 (w_ls を非負化し、合計 = 1 にする)
        double sum = 0.0;
        for(int i = 0; i < n; i++) {
            if(w_ls[i] < 0.0) w_ls[i] = 0.0;
            sum += w_ls[i];
        }
        if(sum <= 0.0) {
            // 全部 0 になったら一様分配
            double uni = 1.0 / n;
            for(int i = 0; i < n; i++) {
                w_ls[i] = uni;
            }
        } else {
            for(int i = 0; i < n; i++) {
                w_ls[i] /= sum;
            }
        }

        // 3) b = A_S^T * t を計算
        vector<double> b(n, 0.0);
        for(int i = 0; i < n; i++) {
            int pk = indices[i];
            b[i] = paints[pk][0] * t[0] + paints[pk][1] * t[1] + paints[pk][2] * t[2];
        }

        // 4) w^T G w と -2 b^T w を計算
        double wGw = 0.0;
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < n; j++) {
                wGw += w_ls[i] * Gram[i][j] * w_ls[j];
            }
        }
        double bTw = 0.0;
        for(int i = 0; i < n; i++) {
            bTw += b[i] * w_ls[i];
        }
        double eprime = wGw - 2.0 * bTw;

        // 5) 二乗誤差 = eprime + t_norm2
        double true_err = eprime + t_norm2;

        return Result{true_err, indices, w_ls};
    }

    void invertMatrix(int size) const {
        // tmp は size × (2*size) の拡大行列 [G | I]
        vector<vector<double>> tmp(size, vector<double>(2 * size, 0.0));
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                tmp[i][j] = Gram[i][j];
            }
            for(int j = 0; j < size; j++) {
                tmp[i][size + j] = (i == j ? 1.0 : 0.0);
            }
        }
        // Gauss-Jordan
        for(int i = 0; i < size; i++) {
            // ピボット選択
            int pivot = i;
            for(int r = i + 1; r < size; r++) {
                if(fabs(tmp[r][i]) > fabs(tmp[pivot][i])) {
                    pivot = r;
                }
            }
            if(pivot != i) {
                swap(tmp[i], tmp[pivot]);
            }
            double diag = tmp[i][i];
            if(fabs(diag) < 1e-12) {
                diag = (diag >= 0 ? 1e-12 : -1e-12);
                tmp[i][i] = diag;
            }
            double invDiag = 1.0 / tmp[i][i];
            for(int c = 0; c < 2 * size; c++) {
                tmp[i][c] *= invDiag;
            }
            for(int r = 0; r < size; r++) {
                if(r == i) continue;
                double factor = tmp[r][i];
                if(fabs(factor) < 1e-16) continue;
                for(int c = 0; c < 2 * size; c++) {
                    tmp[r][c] -= factor * tmp[i][c];
                }
            }
        }
        // 右半分が逆行列
        invG.assign(size, vector<double>(size, 0.0));
        for(int i = 0; i < size; i++) {
            for(int j = 0; j < size; j++) {
                invG[i][j] = tmp[i][size + j];
            }
        }
    }

    void calc_gram_inv(vector<int> const& comb) {
        int size = static_cast<int>(comb.size());

        Gram.assign(size, vector<double>(size, 0.0));
        for(int i = 0; i < size; i++) {
            for(int j = i; j < size; j++) {
                double dot = paints[comb[i]][0] * paints[comb[j]][0] + paints[comb[i]][1] * paints[comb[j]][1] + paints[comb[i]][2] * paints[comb[j]][2];
                Gram[i][j] = dot;
                if(i != j) Gram[j][i] = dot;
            }
        }

        // Gram の逆行列 invG を計算
        invertMatrix(size);

        // 擬似逆行列 = invG × A_S^T (sz×3)
        pseudo.assign(size, vector<double>(3, 0.0));
        for(int i = 0; i < size; i++) {
            for(int d = 0; d < 3; d++) {
                double sum = 0.0;
                for(int j = 0; j < size; j++) {
                    sum += invG[i][j] * paints[comb[j]][d];
                }
                pseudo[i][d] = sum;
            }
        }
    }
};
// Skipped: utils.hpp already included

// ============================================================================
// 定義
// ============================================================================

const double MAX_TIME = 2800.0;

const int INIT_PARTITION_POS = 1; // パーティション初期値

// const int TOP_N = 10000;
// const int MAX_RESULT = 20;

long long MAX_SIMULATE_CNT = 1e7; // 分数パターンの最大数（目安）

const int BUFFER_TURN = 30; // 30ターンは余裕を持たせる

const double SWITH_POLICY_OBJ_TURN = 10.0;

const int COMMON_MAX_COMB_SIZE = 6;
map<int, int> MAX_COMB_SIZES = {
    {11, 5}, {12, 5}, {13, 5}, {14, 4}, {15, 4}, {16, 4}, {17, 4}, {18, 4}, {19, 4}, {20, 4},
};
const int SEARCH_NUM = 13;

// ============================================================================
// Main
// ============================================================================

struct GroupInfo {
    int k;
    int row_num;
    int start_x;
    std::vector<std::pair<int, int>> roots;
    int now_pos;
    int size;
};

class ManageGroupInfo {
  private:
    int n;
    int k;
    int original_k;
    int init_pos;
    std::vector<GroupInfo> infos;

    std::vector<std::pair<int, int>> create_root(int x, int row_num) {
        std::vector<std::pair<int, int>> roots;
        roots.emplace_back(n - 1, x);

        for(int r = 0; r < row_num; ++r) {
            if(r % 2 == 0) {
                for(int i = n - 2; i >= 0; --i) {
                    roots.emplace_back(i, x + r);
                }
            } else {
                for(int i = 0; i < n - 1; ++i) {
                    roots.emplace_back(i, x + r);
                }
            }
        }

        std::reverse(roots.begin(), roots.end());
        return roots;
    }

    std::vector<GroupInfo> construct_group_info() {
        std::vector<int> num_list(k, 1);
        for(int i = 0; i < n - k; ++i) {
            num_list[0] += 1;
            std::sort(num_list.begin(), num_list.end());
        }

        std::vector<int> acc_num_list(k, 0);
        for(int i = 0; i < k - 1; ++i) {
            acc_num_list[i + 1] = num_list[i] + acc_num_list[i];
        }

        std::vector<GroupInfo> result;
        result.reserve(k);
        for(int ki = 0; ki < k; ++ki) {
            int row_num = num_list[ki];
            int start_x = acc_num_list[ki];
            auto roots = create_root(start_x, row_num);

            GroupInfo info;
            info.k = ki;
            info.row_num = row_num;
            info.start_x = start_x;
            info.roots = std::move(roots);
            info.now_pos = init_pos;
            info.size = static_cast<int>(info.roots.size()) - 1;
            result.push_back(std::move(info));
        }

        return result;
    }

  public:
    ManageGroupInfo(int n_, int k_, int original_k_, int init_pos_ = 2) : n(n_), k(k_), original_k(original_k_), init_pos(init_pos_) {
        infos = construct_group_info();
    }

    int get_start_x(int k_index) const {
        return infos[k_index].start_x;
    }

    int get_now_pos(int k_index) const {
        return infos[k_index].now_pos;
    }

    int get_size(int k_index) const {
        return infos[k_index].size;
    }
    std::tuple<int, int, int, int> get_partition_pos(int k_index, int num) const {
        assert(0 < num && num <= infos[k_index].size);
        auto [y1, x1] = infos[k_index].roots[num - 1];
        auto [y2, x2] = infos[k_index].roots[num];
        return {y1, x1, y2, x2};
    }

    void change_now_pos(int k_index, int pos) {
        infos[k_index].now_pos = pos;
    }

    Action get_toggle_action(int k_index, int num) const {
        auto [y1, x1, y2, x2] = this->get_partition_pos(k_index, num);
        return Action::Toggle(y1, x1, y2, x2);
    }

    Action get_add_paint_action(int k_index) const {
        auto [y, x] = infos[k_index].roots[0];
        return Action::Add(y, x, k_index % this->original_k);
    }

    Action get_deliver_paint_action(int k_index) const {
        auto [y, x] = infos[k_index].roots[0];
        return Action::Deliver(y, x);
    }

    Paint get_paint(int k_index, const State &state) const {
        auto [y, x] = infos[k_index].roots[0];
        auto paint = state.get_paint(y, x);
        return paint;
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
            wall_h[input_data.N - 2][x] = true;
        }

        // ルート内の仕切りを外す
        for(int k : range(input_data.K)) {
            const int root_size = (int)infos[k].roots.size();
            for(int i : range(1, root_size)) {
                auto [y1, x1] = infos[k].roots[i - 1];
                auto [y2, x2] = infos[k].roots[i];
                if(y1 == y2) {
                    wall_v[y1][min(x1, x2)] = false;
                } else {
                    wall_h[min(y1, y2)][x1] = false;
                }
            }
        }

        // 初期のパーティション位置を設定
        for(int k : range(input_data.K)) {
            auto [y1, x1, y2, x2] = this->get_partition_pos(k, this->init_pos);
            if(y1 == y2) {
                wall_v[y1][min(x1, x2)] = true;
            } else {
                wall_h[min(y1, y2)][x1] = true;
            }
        }

        // 混合する仕切りを開けておく
        for(int k : range(input_data.K)) {
            int s = this->get_size(k);
            auto [y1, x1, y2, x2] = this->get_partition_pos(k, s);
            assert(x1 == x2);
            wall_h[min(y1, y2)][x1] = false;
        }

        return Wall(wall_h, wall_v);
    }
};

class FractorManager {
    using KEY = tuple<int, int, int>; // (init_pos, max_denom, apply_frac_cnt)

  private:
    unordered_map<KEY, vector<Fractors>> fractor_map;
    unordered_map<KEY, vector<double>> rates_map;

  public:
    FractorManager(vector<int> &max_denominators) {
        for(auto &max_denom : max_denominators) {
            this->construct(max_denom, 1);
            this->construct(max_denom, 2);
        }
    }

    double calc_rate(const Fractors &fractors) const {
        double rate = 1.0;
        for(auto &fractor : fractors) {
            if(fractor.first == -1) {
                rate = 0.0;
            } else {
                rate *= (double)(fractor.first) / (double)(fractor.second);
            }
        }
        return rate;
    }

    void construct(int max_denom, int apply_frac_cnt) {
        vector<Fractors> fractors;
        vector<double> rates;

        // 全開放
        fractors.push_back({make_pair(1, 1)});
        rates.push_back(1.0);

        // 何もしない
        fractors.push_back({make_pair(-1, -1)});
        rates.push_back(0.0);

        // !INFO
        // 分数の適応パターンが多すぎる場合、パターンの列挙だけでTLEする可能性がある。
        // 仕方なくパターン数を減らすが、もっと良い方法があるかもしれない。

        int MAX_STEP = 1;
        long long simulate_cnt = pow(max_denom, 5);
        if(simulate_cnt > MAX_SIMULATE_CNT) {
            double div = (double)simulate_cnt / (double)MAX_SIMULATE_CNT;
            MAX_STEP = max(1, int(round(pow(div, 1.0 / 5.0))));
        }

        unordered_set<Fractor> fractor_set;
        for(int init_pos : range(max_denom, 0, -1)) {
            for(int fractor_cnt : range(apply_frac_cnt)) {
                // 分数1回適応
                if(fractor_cnt == 0) {
                    for(int denominator : range(init_pos, max_denom + 1)) {
                        for(int numerator : range(1, denominator)) {
                            Fractor fractor = make_pair(numerator, denominator);
                            Fractor reduced_fractor = reduce_fraction(fractor);
                            if(fractor_set.contains(reduced_fractor)) continue;
                            fractor_set.insert(reduced_fractor);
                            fractors.push_back({fractor});
                            rates.emplace_back(calc_rate(fractors.back()));
                        }
                    }
                } else if(fractor_cnt == 1) {
                    // 分数2回適応
                    for(int d1 : range(init_pos + 1, max_denom + 1, MAX_STEP)) {
                        for(int n1 : range(1, d1, MAX_STEP)) {
                            Fractor f1 = make_pair(n1, d1);
                            // 次の分母の最小値 = 下側のブロック数 = 前の分子
                            // 次の分母の最大値 += 最大ブロック数 - 前の分母
                            for(int d2 : range(max(2, n1), n1 + (max_denom - d1) + 1, MAX_STEP)) {
                                for(int n2 : range(1, d2, MAX_STEP)) {
                                    Fractor f2 = make_pair(n2, d2);
                                    Fractor fractor = mul_fracs({f1, f2});
                                    if(fractor_set.contains(fractor)) continue;
                                    fractor_set.insert(fractor);
                                    fractors.push_back({f1, f2});
                                    rates.emplace_back(calc_rate(fractors.back()));
                                }
                            }
                        }
                    }
                }
            }

            // 2分探索のためソートしておく必要がある
            auto inds = make_sorted_indices(rates);
            reorder_vector(fractors, inds);
            reorder_vector(rates, inds);

            KEY key = make_tuple(init_pos, max_denom, apply_frac_cnt);
            fractor_map[key] = fractors;
            rates_map[key] = rates;
        }
    }

    const vector<Fractors> &get_fractors(int pos, int max_denom, int apply_frac_cnt) const {
        KEY key = make_tuple(pos, max_denom, apply_frac_cnt);
        return fractor_map.at(key);
    }

    const vector<double> &get_rates(int pos, int max_denom, int apply_frac_cnt) const {
        KEY key = make_tuple(pos, max_denom, apply_frac_cnt);
        return rates_map.at(key);
    }

    pair<double, Fractors> get(int pos, int max_denom, int apply_frac_cnt, int i) const {
        KEY key = make_tuple(pos, max_denom, apply_frac_cnt);
        auto &frac = fractor_map.at(key)[i];
        auto &rate = rates_map.at(key)[i];
        return {rate, frac};
    }
};

void PolicyGreedy(Input &input, State &state, ManageGroupInfo &group_info) {
    auto target_color = input.target[state.deliver_cnt];

    double min_cost = 1e9;
    int min_k = -1;
    bool is_add = false;
    for(int k : range(state.input.K)) {
        Paint paint = group_info.get_paint(k, state);
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
        int now_pos = group_info.get_now_pos(min_k);
        if(now_pos == 1) {
            // 仕切りが1しかない場合は、仕切りを追加する
            const int POS = 2;
            state.apply(group_info.get_toggle_action(min_k, POS));
            state.apply(group_info.get_toggle_action(min_k, now_pos));
            group_info.change_now_pos(min_k, POS);
        }
        state.apply(group_info.get_add_paint_action(min_k));
    }

    state.apply(group_info.get_deliver_paint_action(min_k));
}

struct ImmediateInfo {
    int k;
    bool is_add;
    double rate;
    double vol;
    Fractors fractors;
};

struct DicisionAction {
    vector<Action> pre_actions;
    vector<Action> release_actions;
    vector<Action> post_actions;
    int act_cnt;
    int change_color_num;
};

class DicisionActionPerResult {
  public:
    FractorManager &fractor_manager;
    ManageGroupInfo &manage_group_info;
    Input &input;
    State &state;

    DicisionActionPerResult(FractorManager &fractor_manager_, ManageGroupInfo &manage_group_info_, Input &input_, State &state_)
        : fractor_manager(fractor_manager_), manage_group_info(manage_group_info_), input(input_), state(state_) {
    }

    tuple<int, int> search_target_weight_idx(int k, double target_vol, bool is_add, int max_mul_cnt) {
        double now_vol = manage_group_info.get_paint(k, this->state).vol;
        int now_pos = manage_group_info.get_now_pos(k);
        int max_group_size = manage_group_info.get_size(k);
        auto &rates = fractor_manager.get_rates(now_pos, max_group_size, max_mul_cnt);

        // ---------------------------------------
        // now_vol * rate = target_vol
        // rate = target_vol / now_vol
        // rate = target_vol / (now_vol + 1.0)
        // ---------------------------------------
        double search_rate;
        if(is_add) {
            search_rate = target_vol / (1.0 + now_vol);
        } else {
            search_rate = target_vol / now_vol;
        }
        auto it = upper_bound(ALL(rates), search_rate);
        int it_ind = distance(rates.begin(), it);
        int rates_size = (int)rates.size();
        if(it_ind >= rates_size) {
            it_ind = rates_size - 1;
        }
        return {it_ind, rates_size};
    }

    tuple<vector<ImmediateInfo>, double> eval_one_result(ColorMixer::Result &constrait, vector<int> &max_frac_cnt) {
        int comb_size = constrait.indices.size();

        // 2^comb_size 個の組み合わせを評価する
        vector<vector<ImmediateInfo>> infos;
        for(int comb_ind : range(comb_size)) {
            auto &k = constrait.indices[comb_ind];
            auto &target_vol = constrait.weights[comb_ind];
            double now_vol = manage_group_info.get_paint(k, state).vol;
            bool is_add = (target_vol > now_vol) ? true : false;
            auto [it_ind, max_ind] = search_target_weight_idx(k, target_vol, is_add, max_frac_cnt[comb_ind]);
            vector<ImmediateInfo> immediate_infos;

            const int SEARCH_LEFT = -1;
            const int SEARCH_RIGHT = 1;
            for(int j : range(SEARCH_LEFT, SEARCH_RIGHT)) {
                int new_ind;
                if(it_ind + j < 0) {
                    new_ind = it_ind; // あえて0にする
                } else {
                    new_ind = it_ind + j;
                }
                auto [rate, fractors] = fractor_manager.get(manage_group_info.get_now_pos(k), manage_group_info.get_size(k), max_frac_cnt[comb_ind], new_ind);
                double vol;
                if(is_add) {
                    vol = (now_vol + 1.0) * rate;
                } else {
                    vol = now_vol * rate;
                }
                ImmediateInfo info = {.k = k, .is_add = is_add, .rate = rate, .vol = vol, .fractors = fractors};
                immediate_infos.emplace_back(info);
            }
            infos.emplace_back(move(immediate_infos));
        }

        // 評価関数
        auto eval_cost = [&](int indices) -> double {
            auto &now_target = this->state.input.target[this->state.deliver_cnt];

            double sum_vol = 0.0;
            int add_cnt = 0;
            vector<double> vols;
            vector<Color> colors;
            for(int i : range(comb_size)) {
                int j = (indices >> i) & 1;
                vols.emplace_back(infos[i][j].vol);
                colors.emplace_back(this->input.own[infos[i][j].k]);
                sum_vol += infos[i][j].vol;
                if(infos[i][j].is_add) add_cnt++;
            }

            Color mixed_color = mix(vols, colors);
            double err_cost = eval_error(mixed_color, now_target) * 1e4;
            double discard_cost = max(0.0, sum_vol - 1.0) * (double)(this->input.D);

            int total_add_cnt = this->state.add_cnt + add_cnt;
            if(total_add_cnt > input.H) {
                double add_cost = (total_add_cnt - input.H) * (double)(this->input.D);
                return err_cost + add_cost;
            } else {
                return err_cost + discard_cost;
            }
        };

        double best_cost = 1e9;
        int best_indices = 0;

        // for(int x : range(1 << comb_size)) {
        //     double sum_vol = 0.0;
        //     for(int i : range(comb_size)) {
        //         int j = (x >> i) & 1;
        //         sum_vol += infos[i][j].vol;
        //     }
        //     if(sum_vol > 1.0 - 1e-6) {
        //         double cost = eval_cost(x);
        //         if(cost < best_cost) {
        //             best_cost = cost;
        //             best_indices = x;
        //         }
        //     }
        // }

        int temp_x = (1 << comb_size) - 1;
        double sum_vol = 0.0;
        for(int i : range(comb_size)) {
            int j = (temp_x >> i) & 1;
            sum_vol += infos[i][j].vol;
        }
        if(sum_vol > 1.0 - 1e-6) {
            double cost = eval_cost(temp_x);
            if(cost < best_cost) {
                best_cost = cost;
                best_indices = temp_x;
            }
        }

        vector<ImmediateInfo> best_info;
        for(int i : range(comb_size)) {
            int j = (best_indices >> i) & 1;
            best_info.emplace_back(infos[i][j]);
        }

        return {best_info, best_cost};
    }
};

DicisionAction construct_from_immediateinfo(vector<ImmediateInfo> &best_info, ManageGroupInfo &manage_group_info) {
    DicisionAction action_result;
    action_result.change_color_num = (int)best_info.size();

    for(auto &info : best_info) {
        int now_partition_pos = manage_group_info.get_now_pos(info.k);
        int frac_size = info.fractors.size();

        auto &first_fractor = info.fractors[0];
        if(first_fractor.first == -1 && first_fractor.second == -1) {
            // 何もしない
            continue;
        } else if(first_fractor.first == 1 && first_fractor.second == 1) {
            // 全開放
            assert(frac_size == 1);
            action_result.release_actions.emplace_back(manage_group_info.get_toggle_action(info.k, now_partition_pos));
            if(info.is_add) {
                // 仕切りを解放してから絵の具追加する
                action_result.pre_actions.emplace_back(manage_group_info.get_add_paint_action(info.k));
            }
            action_result.post_actions.emplace_back(manage_group_info.get_toggle_action(info.k, INIT_PARTITION_POS));
            manage_group_info.change_now_pos(info.k, INIT_PARTITION_POS);
        } else {
            // 分割n回適応
            int upper_partition = 0;
            int lower_partition = now_partition_pos;
            for(int fi : range(frac_size)) {
                auto &fractor = info.fractors[fi];

                // 上の仕切りから、分母だけ進んだのがstopしたいしきり位置
                int stop_par_pos = upper_partition + fractor.second;
                // stopする仕切りから、分子だけ進んだのが、releaseする仕切り位置
                int release_par_pos = stop_par_pos - fractor.first;

                if(stop_par_pos != lower_partition) {
                    // 現在の仕切りを動かす必要があるなら、仕切りを拡張する
                    action_result.pre_actions.emplace_back(manage_group_info.get_toggle_action(info.k, stop_par_pos));
                    action_result.pre_actions.emplace_back(manage_group_info.get_toggle_action(info.k, lower_partition));
                    // 拡張した後に追加する
                    if(fi == 0 && info.is_add) {
                        action_result.pre_actions.emplace_back(manage_group_info.get_add_paint_action(info.k));
                    }
                } else {
                    // 追加する
                    if(fi == 0 && info.is_add) {
                        assert(lower_partition > 1);
                        action_result.pre_actions.emplace_back(manage_group_info.get_add_paint_action(info.k));
                    }
                }
                // 分子の位置で止める
                action_result.pre_actions.emplace_back(manage_group_info.get_toggle_action(info.k, release_par_pos));

                if(fi == frac_size - 1) {
                    // 分母の位置で解放する
                    action_result.release_actions.emplace_back(manage_group_info.get_toggle_action(info.k, stop_par_pos));
                    // 最後の仕切り位置は、release地点になる
                    manage_group_info.change_now_pos(info.k, release_par_pos);
                } else {
                    // 仮止めした仕切りは解放しておく必要がある
                    action_result.post_actions.emplace_back(manage_group_info.get_toggle_action(info.k, release_par_pos));
                }

                upper_partition = release_par_pos;
                lower_partition = stop_par_pos;
            }
        }
    }
    int act_cnt = action_result.pre_actions.size() + action_result.release_actions.size() + action_result.post_actions.size();
    action_result.act_cnt = act_cnt;

    return action_result;
}

DicisionAction dicision_action(Input &input, State &state, ColorMixer &mixer, double obj_turn, ManageGroupInfo &group_info, FractorManager &fractor_manager) {
    Color target = input.target[state.deliver_cnt];
    DicisionActionPerResult per_result = DicisionActionPerResult(fractor_manager, group_info, input, state);

    vector<int> comb_sizes;
    if(MAX_COMB_SIZES.contains(input.K)) {
        for(int i : range(2, min(MAX_COMB_SIZES[input.K], COMMON_MAX_COMB_SIZE) + 1)) {
            comb_sizes.push_back(i);
        }
    } else {
        for(int i : range(2, COMMON_MAX_COMB_SIZE + 1)) {
            comb_sizes.push_back(i);
        }
    }

    double best_cost = 1e9;
    vector<ImmediateInfo> best_info;
    for(int comb_size : comb_sizes) {
        double remain_turn = obj_turn - comb_size * 4.0 - 2.0;
        if(remain_turn < 0.0) {
            continue; // 目標ターン数を超える場合はスキップ
        }

        auto results = mixer.solve_nnls(target, comb_size, SEARCH_NUM);
        for(auto &result : results) {
            int max_double_frac_num = (int)(remain_turn / 4.0); // 分数2回適応できる数
            max_double_frac_num = min(max_double_frac_num, comb_size);
            vector<int> max_frac_cnt(comb_size, 1);
            if(max_double_frac_num > 0) {
                for(int i : range(max_double_frac_num)) {
                    max_frac_cnt[comb_size - i - 1] = 2;
                }
            }
            do {
                auto [now_info, now_cost] = per_result.eval_one_result(result, max_frac_cnt);
                if(now_cost < best_cost) {
                    best_cost = now_cost;
                    best_info = now_info;
                }
            } while(next_permutation(ALL(max_frac_cnt)));
        }
    }

    assert((int)best_info.size() != 0);

    auto action_result = construct_from_immediateinfo(best_info, group_info);
    return action_result;
}

void discard_mix_well(State &state, Input &input) {
    while(state.get_paint(input.N - 1, 0).vol > 1e-6) {
        state.apply(Action::Discard(input.N - 1, 0));
    }
}

void print_info(State &state) {
    auto [deliver_cost, err_cost, total_cost] = state.get_score();
    cerr << boost::format("H: %4d | Turn: %5d/%5d | Add: %4d | Discard: %4d (%5d loss) | Score: %5d (add: %5d, err: %5d)") % state.deliver_cnt % state.turn %
                state.input.T % state.add_cnt % state.discard_cnt % int(state.discard * 1e4) % total_cost % deliver_cost % err_cost
         << endl;
}

void solve() {
    TimeKeeper time_keeper(MAX_TIME);

    Input input = parse_input();
    ManageGroupInfo manage_group_info(input.N, input.K, input.K, INIT_PARTITION_POS);

    set<int> unique_denoms;
    for(int k : range(input.K)) {
        unique_denoms.insert(manage_group_info.get_size(k));
    }
    vector<int> denoms(ALL(unique_denoms));

    FractorManager fractor_manager(denoms);
    auto init_wall = manage_group_info.struct_init_wall(input);
    State state(init_wall, input);
    ColorMixer mixer(input.own);

    // Main Loop
    int policy_greedy_cnt = 0;
    double policy_err_sum = 0.0;
    map<int, int> act_cnt;
    map<int, int> color_cnt;

    try {
        // !DEBUG
        for(int h : range(input.H)) {
            if(h % 10 == 0) print_info(state);
            // print_info(state);

            int remain_turn = input.T - state.turn - BUFFER_TURN;
            double obj_turn = (double)remain_turn / (double)(input.H - state.deliver_cnt);

            if(obj_turn >= SWITH_POLICY_OBJ_TURN) {
                auto action_result = dicision_action(input, state, mixer, obj_turn, manage_group_info, fractor_manager);

                act_cnt[action_result.change_color_num] += action_result.act_cnt;
                color_cnt[action_result.change_color_num]++;

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

            } else {
                auto pre_err = state.error;
                PolicyGreedy(input, state, manage_group_info);
                auto post_err = state.error;
                policy_err_sum += post_err - pre_err;
                policy_greedy_cnt++;
            }
        }
        print_info(state);
    } catch(const exception &e) {
        Output output = Output{init_wall, state.actions};
        print_output(output);
        cerr << "Exception: " << e.what() << endl;
        exit(1);
    }

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
    cerr << boost::format("score: %d, elapsed: %f, turn: %d/%d") % get<2>(state.get_score()) % time_keeper.getElapsedTime() % state.turn % input.T << endl;

    // output
    Output output = Output{init_wall, state.actions};
    print_output(output);
}

int main() {
    solve();
    return 0;
}
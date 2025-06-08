#include <algorithm>
#include <cassert>
#include <iostream>
#include <vector>

/**
 * y: 入力配列 (長さN)
 * s: 合計制約 (0 <= s <= N)
 * x: 出力配列 (長さN, 書き込み)
 * e: 出力値（誤差, optional）
 */
void projection(int N, const double *y, double s, double *x, double *e = nullptr) {
    int i, j;

    if((s < 0) || (s > N)) {
        std::cerr << "impossible sum constraint!\n";
        std::exit(-1);
    }

    if(s == 0) {
        if(e) *e = 0;
        for(i = 0; i < N; i++) {
            x[i] = 0;
            if(e) *e += y[i] * y[i];
        }
        if(e) *e *= 0.5;
        return;
    }

    if(s == N) {
        if(e) *e = 0;
        for(i = 0; i < N; i++) {
            x[i] = 1;
            if(e) *e += (1 - y[i]) * (1 - y[i]);
        }
        if(e) *e *= 0.5;
        return;
    }

    struct mypair {
        double number;
        int index;
        void setval(double n, int i) {
            number = n;
            index = i;
        }
    };
    auto mycompare = [](const mypair &l, const mypair &r) { return l.number < r.number; };

    std::vector<mypair> v(N);
    for(i = 0; i < N; i++)
        v[i].setval(y[i], i);
    std::sort(v.begin(), v.end(), mycompare);

    // Compute partial sums.
    std::vector<double> T(N + 1, 0.0);
    for(i = 1; i <= N; i++)
        T[i] = T[i - 1] + v[i - 1].number;

    double gamma = 0.0;
    bool flag = false;
    for(i = 0; i <= N; i++) {
        // i==j
        if((i + s) == N)
            if((i == 0) || (v[i].number >= v[i - 1].number + 1)) {
                j = i;
                flag = true;
                break;
            }
        // i<j
        for(j = i + 1; j <= N; j++) {
            gamma = (s + j - N + T[i] - T[j]) / (j - i);
            if(i == 0) {
                if(j == N) {
                    if((v[i].number + gamma > 0) && (v[j - 1].number + gamma < 1)) {
                        flag = true;
                        break;
                    }
                } else {
                    if((v[i].number + gamma > 0) && (v[j - 1].number + gamma < 1) && (v[j].number + gamma >= 1)) {
                        flag = true;
                        break;
                    }
                }
            } else if(j == N) {
                if((v[i - 1].number + gamma <= 0) && (v[i].number + gamma > 0) && (v[j - 1].number + gamma < 1)) {
                    flag = true;
                    break;
                }
            } else {
                if((v[i - 1].number + gamma <= 0) && (v[i].number + gamma > 0) && (v[j - 1].number + gamma < 1) && (v[j].number + gamma >= 1)) {
                    flag = true;
                    break;
                }
            }
        }
        if(flag) break;
    }

    if(e) *e = 0;
    int k;
    for(k = 0; k < i; k++) {
        x[v[k].index] = 0;
        if(e) *e += (v[k].number) * (v[k].number);
    }
    for(k = i; k < j; k++) {
        x[v[k].index] = v[k].number + gamma;
        if(e) *e += gamma * gamma;
    }
    for(k = j; k < N; k++) {
        x[v[k].index] = 1;
        if(e) *e += (1 - v[k].number) * (1 - v[k].number);
    }
    if(e) *e *= 0.5;
}

#include <iostream>
#include <vector>

int main() {
    std::vector<double> y = {0.2, 0.9, -0.4, 0.7};
    int N = y.size();
    double s = 2.0;
    std::vector<double> x(N);
    double e;
    projection(N, y.data(), s, x.data(), &e);
    std::cout << "projected x = ";
    for(auto v : x)
        std::cout << v << " ";
    std::cout << "\nerror = " << e << std::endl;
}

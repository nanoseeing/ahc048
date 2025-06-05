#pragma once

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

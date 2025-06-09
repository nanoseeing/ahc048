#ifndef NNLS_ACTIVE_CONSTRAIT_HPP
#define NNLS_ACTIVE_CONSTRAIT_HPP

#include <Eigen/Core>
#include <Eigen/QR>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <vector>

// ----------- 頑健なBox+Simplex射影 ここから ---------------
template <typename DerivedVec>
Eigen::Matrix<typename DerivedVec::Scalar, Eigen::Dynamic, 1> project_boxed_simplex(const Eigen::MatrixBase<DerivedVec> &y,
                                                                                    const std::vector<typename DerivedVec::Scalar> &u, int max_iter = 100,
                                                                                    typename DerivedVec::Scalar tol = 1e-12) {
    using Scalar = typename DerivedVec::Scalar;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
    const int n = y.size();
    assert((int)u.size() == n);

    Scalar lambda_low = std::numeric_limits<Scalar>::lowest();
    Scalar lambda_high = std::numeric_limits<Scalar>::max();
    for(int i = 0; i < n; ++i) {
        lambda_low = std::max(lambda_low, y(i) - u[i]);
        lambda_high = std::min(lambda_high, y(i));
    }

    auto f = [&](Scalar lambda) -> Scalar {
        Scalar sum = 0;
        for(int i = 0; i < n; ++i) {
            Scalar val = std::min(u[i], std::max(Scalar(0), y(i) - lambda));
            sum += val;
        }
        return sum;
    };

    Scalar target = Scalar(1);
    Scalar l = lambda_low - 1, r = lambda_high + 1; // 余裕を持たせる
    for(int iter = 0; iter < max_iter; ++iter) {
        Scalar m = (l + r) / 2;
        Scalar s = f(m);
        if(s > target)
            l = m;
        else
            r = m;
        if(std::abs(r - l) < tol) break;
    }
    Scalar lambda_star = (l + r) / 2;

    Vector x(n);
    for(int i = 0; i < n; ++i) {
        x(i) = std::min(u[i], std::max(Scalar(0), y(i) - lambda_star));
    }
    // 念のため再正規化
    Scalar sum_x = x.sum();
    if(std::abs(sum_x - 1) > tol && sum_x > tol) {
        Scalar diff = sum_x - 1;
        for(int i = 0; i < n; ++i)
            x(i) -= diff / n;
        for(int i = 0; i < n; ++i)
            x(i) = std::min(u[i], std::max(Scalar(0), x(i)));
    }
    return x;
}
// ----------- 頑健なBox+Simplex射影 ここまで ---------------

namespace Eigen {

/*
 * BVLS_PGD_SumOne: Box‐Constrained Least Squares with ∑v_i = 1
 * via Projected Gradient Descent.
 */
template <class MatrixType_>
class BVLS_PGD_SumOne {
  public:
    typedef MatrixType_ MatrixType;
    typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;
    typedef Matrix<Scalar, MatrixType::ColsAtCompileTime, 1> VectorType;
    typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> RhsVectorType;

    BVLS_PGD_SumOne() = default;

    BVLS_PGD_SumOne(const MatrixType &A, const std::vector<Scalar> &u, Index max_iter = 1000, Scalar step_size = Scalar(1e-3), Scalar tol = Scalar(1e-8))
        : A_(A), K_(A.cols()), max_iter_(max_iter), alpha_(step_size), tolerance_(tol) {
        assert((Index)u.size() == K_ && "u.size() must equal A.cols()");
        upperBounds_ = u;
    }

    template <typename MatrixDerived>
    BVLS_PGD_SumOne<MatrixType> &compute(const EigenBase<MatrixDerived> &A, const std::vector<Scalar> &u) {
        A_ = A.derived();
        K_ = A_.cols();
        assert((Index)u.size() == K_ && "u.size() must equal A.cols()");
        upperBounds_ = u;
        return *this;
    }

    VectorType solve(const RhsVectorType &b) {
        assert((Index)b.size() == A_.rows() && "b.size() must match A.rows()");
        const Index n = K_;
        // 初期解: 均等分配
        VectorType v = VectorType::Constant(n, Scalar(1) / Scalar(n));
        for(Index i = 0; i < n; ++i) {
            if(v(i) > upperBounds_[i]) v(i) = upperBounds_[i];
        }
        // 均等化後、Box+Simplexに射影
        v = project_boxed_simplex(v, upperBounds_);

        for(Index iter = 0; iter < max_iter_; ++iter) {
            // 1) 勾配: g = 2 A^T (A v - b)
            VectorType g = A_.transpose() * (A_ * v - b);
            g *= Scalar(2.0);

            // 2) 勾配降下
            VectorType y = v - alpha_ * g;

            // 3) 射影
            VectorType v_next = project_boxed_simplex(y, upperBounds_);

            // 4) 収束判定
            if((v_next - v).norm() < tolerance_) {
                v = v_next;
                break;
            }
            v = v_next;
        }
        return v;
    }

    BVLS_PGD_SumOne<MatrixType> &setStepSize(Scalar a) {
        alpha_ = a;
        return *this;
    }
    BVLS_PGD_SumOne<MatrixType> &setMaxIterations(Index m) {
        max_iter_ = m;
        return *this;
    }
    BVLS_PGD_SumOne<MatrixType> &setTolerance(Scalar tol) {
        tolerance_ = tol;
        return *this;
    }

  private:
    MatrixType A_;
    Index K_ = 0;
    std::vector<Scalar> upperBounds_;
    Index max_iter_ = 1000;
    Scalar alpha_ = Scalar(1e-3);
    Scalar tolerance_ = Scalar(1e-8);
};

} // namespace Eigen

#endif // NNLS_ACTIVE_CONSTRAIT_HPP

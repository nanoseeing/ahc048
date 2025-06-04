namespace Eigen {
namespace internal {

template <typename MatrixQR, typename HCoeffs, typename VectorQR>
void householder_qr_inplace_update(MatrixQR& mat, HCoeffs& hCoeffs, const VectorQR& newColumn, typename MatrixQR::Index k,
                                   typename MatrixQR::Scalar* tempData) {
    typedef typename MatrixQR::Index Index;
    typedef typename MatrixQR::RealScalar RealScalar;
    Index rows = mat.rows();

    eigen_assert(k < mat.cols());
    eigen_assert(k < rows);
    eigen_assert(hCoeffs.size() == mat.cols());
    eigen_assert(newColumn.size() == rows);
    eigen_assert(tempData);

    // Store new column in mat at column k
    mat.col(k) = newColumn;
    // Apply H = H_1...H_{k-1} on newColumn (skip if k=0)
    for(Index i = 0; i < k; ++i) {
        Index remainingRows = rows - i;
        mat.col(k).tail(remainingRows).applyHouseholderOnTheLeft(mat.col(i).tail(remainingRows - 1), hCoeffs.coeffRef(i), tempData + i + 1);
    }
    // Construct Householder projector in-place in column k
    RealScalar beta;
    mat.col(k).tail(rows - k).makeHouseholderInPlace(hCoeffs.coeffRef(k), beta);
    mat.coeffRef(k, k) = beta;
}

} // namespace internal
} // namespace Eigen

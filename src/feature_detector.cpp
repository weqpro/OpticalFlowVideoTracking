// Copyright 2026 Konovalenko Stanislav and Hombosh Oleh
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "feature_detector.h"
#include "optical_flow.h"
#include <algorithm>
#include <cmath>

namespace vision {

void computeSpatialGradients(const Eigen::MatrixXd& image, Eigen::MatrixXd& grad_ix, Eigen::MatrixXd& grad_iy) {
    const int ROWS_COUNT = static_cast<int>(image.rows());
    const int COLS_COUNT = static_cast<int>(image.cols());
    grad_ix.setZero(ROWS_COUNT, COLS_COUNT);
    grad_iy.setZero(ROWS_COUNT, COLS_COUNT);
    
    Eigen::Matrix3d kernel_x;
    Eigen::Matrix3d kernel_y;
    kernel_x << -3, 0, 3, -10, 0, 10, -3, 0, 3;
    kernel_y << -3, -10, -3, 0, 0, 0, 3, 10, 3;

    for (int row_idx = 1; row_idx < ROWS_COUNT - 1; ++row_idx) {
        for (int col_idx = 1; col_idx < COLS_COUNT - 1; ++col_idx) {
            const auto REGION = image.block<3, 3>(row_idx - 1, col_idx - 1).array();
            grad_ix(row_idx, col_idx) = (REGION * kernel_x.array()).sum();
            grad_iy(row_idx, col_idx) = (REGION * kernel_y.array()).sum();
        }
    }
}

Eigen::MatrixXd computeMinEigenvalueMap(const Eigen::MatrixXd& grad_ix, const Eigen::MatrixXd& grad_iy) {
    const int ROWS_COUNT = static_cast<int>(grad_ix.rows());
    const int COLS_COUNT = static_cast<int>(grad_ix.cols());
    
    const Eigen::MatrixXd IXX_MAP = grad_ix.array().square();
    const Eigen::MatrixXd IYY_MAP = grad_iy.array().square();
    const Eigen::MatrixXd IXY_MAP = grad_ix.array() * grad_iy.array();
    Eigen::MatrixXd eig_min = Eigen::MatrixXd::Zero(ROWS_COUNT, COLS_COUNT);

    for (int row_idx = 1; row_idx < ROWS_COUNT - 1; ++row_idx) {
        for (int col_idx = 1; col_idx < COLS_COUNT - 1; ++col_idx) {
            const double SUM_XX = IXX_MAP.block<3, 3>(row_idx - 1, col_idx - 1).sum();
            const double SUM_YY = IYY_MAP.block<3, 3>(row_idx - 1, col_idx - 1).sum();
            const double SUM_XY = IXY_MAP.block<3, 3>(row_idx - 1, col_idx - 1).sum();
            eig_min(row_idx, col_idx) = 0.5 * (SUM_XX + SUM_YY - std::sqrt((SUM_XX - SUM_YY) * (SUM_XX - SUM_YY) + 4.0 * SUM_XY * SUM_XY));
        }
    }
    return eig_min;
}

static bool isPeak(const Eigen::MatrixXd& eig_min, const int ROW_IDX, const int COL_IDX, const double VAL) {
    for (int ni_idx = -1; ni_idx <= 1; ++ni_idx) {
        for (int nj_idx = -1; nj_idx <= 1; ++nj_idx) {
            if (ni_idx == 0 && nj_idx == 0) {
                continue;
            }
            if (eig_min(ROW_IDX + ni_idx, COL_IDX + nj_idx) > VAL) {
                return false;
            }
        }
    }
    return true;
}

std::vector<CornerCandidate> collectLocalMaxima(const Eigen::MatrixXd& eig_min, const double THRESHOLD) {
    const int ROWS_COUNT = static_cast<int>(eig_min.rows());
    const int COLS_COUNT = static_cast<int>(eig_min.cols());
    std::vector<CornerCandidate> candidates;

    for (int row_idx = 1; row_idx < ROWS_COUNT - 1; ++row_idx) {
        for (int col_idx = 1; col_idx < COLS_COUNT - 1; ++col_idx) {
            const double VAL = eig_min(row_idx, col_idx);
            if (VAL > THRESHOLD && isPeak(eig_min, row_idx, col_idx, VAL)) {
                candidates.push_back({row_idx, col_idx, VAL});
            }
        }
    }
    return candidates;
}

static void refineCorner(const Eigen::MatrixXd& image, Eigen::Vector2d& corner) {
    const int MAX_ITER = 5;
    for (int iter_idx = 0; iter_idx < MAX_ITER; ++iter_idx) {
        double current_grad_x = 0.0;
        double current_grad_y = 0.0;
        double current_grad_t = 0.0;
        computePixelGradients(image, image, corner, corner, current_grad_x, current_grad_y, current_grad_t);
        
        const Eigen::Vector2d DELTA_VEC(current_grad_x, current_grad_y);
        if (DELTA_VEC.norm() < 0.01) {
            break;
        }
        corner += DELTA_VEC * 0.1; 
    }
}

std::vector<Eigen::Vector2d> findGoodFeaturesToTrack(
    const Eigen::MatrixXd& image,
    const int MAX_CORNERS,
    const double QUALITY_LEVEL,
    const double MIN_DISTANCE
) {
    if (image.rows() < 3 || image.cols() < 3) {
        return {};
    }

    Eigen::MatrixXd grad_ix;
    Eigen::MatrixXd grad_iy;
    computeSpatialGradients(image, grad_ix, grad_iy);
    
    const Eigen::MatrixXd EIG_MAP = computeMinEigenvalueMap(grad_ix, grad_iy);
    
    const double THRESHOLD_VAL = EIG_MAP.maxCoeff() * QUALITY_LEVEL;
    auto candidates = collectLocalMaxima(EIG_MAP, THRESHOLD_VAL);

    std::sort(candidates.begin(), candidates.end(), [](const CornerCandidate& lhs, const CornerCandidate& rhs) {
        return lhs.val > rhs.val;
    });

    std::vector<Eigen::Vector2d> corners;
    for (const auto& cand : candidates) {
        if (corners.size() >= static_cast<size_t>(MAX_CORNERS)) {
            break;
        }
        
        Eigen::Vector2d pos(static_cast<double>(cand.col), static_cast<double>(cand.row));
        bool far_enough = true;
        for (const auto& existing : corners) {
            if ((pos - existing).norm() < MIN_DISTANCE) {
                far_enough = false;
                break;
            }
        }
        if (far_enough) {
            refineCorner(image, pos);
            corners.push_back(pos);
        }
    }
    return corners;
}

} // namespace vision

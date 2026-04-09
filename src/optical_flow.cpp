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

#include "optical_flow.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include "deriv.h"

namespace vision {

// --- Public API Helpers ---

double bilinearInterpolation(const Eigen::MatrixXd& mat, const double x_coord, const double y_coord) {
    const int X_BASE = static_cast<int>(std::floor(x_coord));
    const int Y_BASE = static_cast<int>(std::floor(y_coord));

    if (X_BASE < 0 || Y_BASE < 0 || X_BASE + 1 >= mat.cols() || Y_BASE + 1 >= mat.rows()) {
        return mat(std::clamp(Y_BASE, 0, static_cast<int>(mat.rows() - 1)), 
                   std::clamp(X_BASE, 0, static_cast<int>(mat.cols() - 1)));
    }

    const double DELTA_X = x_coord - static_cast<double>(X_BASE);
    const double DELTA_Y = y_coord - static_cast<double>(Y_BASE);
    
    const double VAL00 = mat(Y_BASE, X_BASE);         
    const double VAL10 = mat(Y_BASE, X_BASE + 1);     
    const double VAL01 = mat(Y_BASE + 1, X_BASE);     
    const double VAL11 = mat(Y_BASE + 1, X_BASE + 1); 
    
    return (VAL00 * (1.0 - DELTA_X) * (1.0 - DELTA_Y) +
            VAL10 * DELTA_X * (1.0 - DELTA_Y) +
            VAL01 * (1.0 - DELTA_X) * DELTA_Y +
            VAL11 * DELTA_X * DELTA_Y);
}

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

static bool isPeak(const Eigen::MatrixXd& eig_min, const int row_idx, const int col_idx, const double val) {
    for (int ni = -1; ni <= 1; ++ni) {
        for (int nj = -1; nj <= 1; ++nj) {
            if (ni == 0 && nj == 0) {
                continue;
            }
            if (eig_min(row_idx + ni, col_idx + nj) > val) {
                return false;
            }
        }
    }
    return true;
}

std::vector<CornerCandidate> collectLocalMaxima(const Eigen::MatrixXd& eig_min, const double threshold) {
    const int ROWS_COUNT = static_cast<int>(eig_min.rows());
    const int COLS_COUNT = static_cast<int>(eig_min.cols());
    std::vector<CornerCandidate> candidates;

    for (int row_idx = 1; row_idx < ROWS_COUNT - 1; ++row_idx) {
        for (int col_idx = 1; col_idx < COLS_COUNT - 1; ++col_idx) {
            const double VAL = eig_min(row_idx, col_idx);
            if (VAL > threshold && isPeak(eig_min, row_idx, col_idx, VAL)) {
                candidates.push_back({row_idx, col_idx, VAL});
            }
        }
    }
    return candidates;
}

void computePixelGradients(
    const Eigen::MatrixXd& img_prev, const Eigen::MatrixXd& img_next,
    const Eigen::Vector2d& prev_pos, const Eigen::Vector2d& next_pos,
    double& grad_x, double& grad_y, double& grad_t
) {
    const double P_X = prev_pos.x();
    const double P_Y = prev_pos.y();
    const double N_X = next_pos.x();
    const double N_Y = next_pos.y();

    const double I1_X = (bilinearInterpolation(img_prev, P_X + 1.0, P_Y) - bilinearInterpolation(img_prev, P_X - 1.0, P_Y)) * 0.5;
    const double I1_Y = (bilinearInterpolation(img_prev, P_X, P_Y + 1.0) - bilinearInterpolation(img_prev, P_X, P_Y - 1.0)) * 0.5;
    const double I2_X = (bilinearInterpolation(img_next, N_X + 1.0, N_Y) - bilinearInterpolation(img_next, N_X - 1.0, N_Y)) * 0.5;
    const double I2_Y = (bilinearInterpolation(img_next, N_X, N_Y + 1.0) - bilinearInterpolation(img_next, N_X, N_Y - 1.0)) * 0.5;
    
    grad_x = (I1_X + I2_X) * 0.5;
    grad_y = (I1_Y + I2_Y) * 0.5;
    grad_t = bilinearInterpolation(img_next, N_X, N_Y) - bilinearInterpolation(img_prev, P_X, P_Y);
}

// --- Public API Implementation ---

static void refineCorner(const Eigen::MatrixXd& image, Eigen::Vector2d& corner) {
    const int MAX_ITER = 5;
    for (int iter = 0; iter < MAX_ITER; ++iter) {
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
    const int max_corners,
    const double quality_level,
    const double min_distance
) {
    if (image.rows() < 3 || image.cols() < 3) {
        return {};
    }

    Eigen::MatrixXd grad_ix;
    Eigen::MatrixXd grad_iy;
    computeSpatialGradients(image, grad_ix, grad_iy);
    
    const Eigen::MatrixXd EIG_MAP = computeMinEigenvalueMap(grad_ix, grad_iy);
    
    const double THRESHOLD_VAL = EIG_MAP.maxCoeff() * quality_level;
    auto candidates = collectLocalMaxima(EIG_MAP, THRESHOLD_VAL);

    std::sort(candidates.begin(), candidates.end(), [](const CornerCandidate& lhs, const CornerCandidate& rhs) {
        return lhs.val > rhs.val;
    });

    std::vector<Eigen::Vector2d> corners;
    for (const auto& cand : candidates) {
        if (corners.size() >= static_cast<size_t>(max_corners)) {
            break;
        }
        
        Eigen::Vector2d pos(static_cast<double>(cand.c), static_cast<double>(cand.r));
        bool far_enough = true;
        for (const auto& existing : corners) {
            if ((pos - existing).norm() < min_distance) {
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

static int solveLKIteration(
    const Eigen::MatrixXd& img_prev,
    const Eigen::MatrixXd& img_next,
    const double center_x, 
    const double center_y,
    double& flow_dx,
    double& flow_dy,
    const int neighborhood_size
) {
    const int HALF_WIN = neighborhood_size / 2;
    const int NUM_ELEMENTS = neighborhood_size * neighborhood_size;
    Eigen::MatrixXd design_matrix(NUM_ELEMENTS, 2);
    Eigen::VectorXd observation_vector(NUM_ELEMENTS);

    for(int row_offset = -HALF_WIN; row_offset <= HALF_WIN; ++row_offset) {
        for(int col_offset = -HALF_WIN; col_offset <= HALF_WIN; ++col_offset) {
            const int IDX_VAL = (row_offset + HALF_WIN) * neighborhood_size + (col_offset + HALF_WIN);
            const Eigen::Vector2d P_POS(center_x + static_cast<double>(col_offset), center_y + static_cast<double>(row_offset));
            const Eigen::Vector2d N_POS(P_POS.x() + flow_dx, P_POS.y() + flow_dy);

            if (P_POS.x() < 1.0 || P_POS.y() < 1.0 || P_POS.x() >= static_cast<double>(img_prev.cols()) - 1.0 || P_POS.y() >= static_cast<double>(img_prev.rows()) - 1.0 ||
                N_POS.x() < 1.0 || N_POS.y() < 1.0 || N_POS.x() >= static_cast<double>(img_next.cols()) - 1.0 || N_POS.y() >= static_cast<double>(img_next.rows()) - 1.0) {
                return -1;
            }

            double current_gx = 0.0;
            double current_gy = 0.0;
            double current_gt = 0.0;
            computePixelGradients(img_prev, img_next, P_POS, N_POS, current_gx, current_gy, current_gt);

            design_matrix(IDX_VAL, 0) = current_gx;
            design_matrix(IDX_VAL, 1) = current_gy;
            observation_vector(IDX_VAL) = -current_gt;
        }
    }

    const Eigen::Matrix2d HESSIAN_MAT = design_matrix.transpose() * design_matrix;
    if (std::abs(HESSIAN_MAT.determinant()) < 1e-9) {
        return -1;
    }

    const Eigen::Vector2d DELTA_VEC = HESSIAN_MAT.ldlt().solve(design_matrix.transpose() * observation_vector);
    flow_dx += DELTA_VEC.x();
    flow_dy += DELTA_VEC.y();
    
    return (DELTA_VEC.norm() < 0.001) ? 0 : 1;
}

void calcOpticalFlowLK(
    const Eigen::MatrixXd& img_prev,
    const Eigen::MatrixXd& img_next,
    std::vector<TrackedFeature>& features,
    const int neighborhood_size,
    const int num_levels
) {
    const int MAX_ITERATIONS = 10;
    const std::vector<Eigen::MatrixXd> PYR_PREV = buildGaussianPyramid(img_prev, num_levels);
    const std::vector<Eigen::MatrixXd> PYR_NEXT = buildGaussianPyramid(img_next, num_levels);
    const int ACTUAL_LEVELS = static_cast<int>(PYR_PREV.size());

    for (auto& feat : features) {
        if (feat.is_lost) {
            continue;
        }

        double flow_dx = 0.0;
        double flow_dy = 0.0;
        bool tracking_failed = false;

        for (int level_idx = ACTUAL_LEVELS - 1; level_idx >= 0; --level_idx) {
            const double SCALE = std::pow(2.0, level_idx);
            const double POS_X = feat.previous_pos.x() / SCALE;
            const double POS_Y = feat.previous_pos.y() / SCALE;

            // Solve LK at current level
            for (int iter = 0; iter < MAX_ITERATIONS; ++iter) {
                const int RES_STATUS = solveLKIteration(PYR_PREV[level_idx], PYR_NEXT[level_idx], POS_X, POS_Y, flow_dx, flow_dy, neighborhood_size);
                if (RES_STATUS == 0) {
                    break; 
                }
                if (RES_STATUS == -1) {
                    tracking_failed = true;
                    break; 
                }
            }

            if (tracking_failed) {
                break;
            }

            // Propagate to finer level
            if (level_idx > 0) {
                flow_dx *= 2.0;
                flow_dy *= 2.0;
            }
        }

        const double FINAL_X = feat.previous_pos.x() + flow_dx;
        const double FINAL_Y = feat.previous_pos.y() + flow_dy;
        
        if (FINAL_X < 0.0 || FINAL_Y < 0.0 || FINAL_X >= static_cast<double>(img_prev.cols()) - 1.0 || FINAL_Y >= static_cast<double>(img_prev.rows()) - 1.0) {
            tracking_failed = true;
        }

        if (!tracking_failed) {
            feat.current_pos = Eigen::Vector2d(FINAL_X, FINAL_Y);
        } else {
            feat.is_lost = true;
        }
    }
}

std::vector<Eigen::MatrixXd> buildGaussianPyramid(const Eigen::MatrixXd& img, const int levels) {
    std::vector<Eigen::MatrixXd> pyramid;
    pyramid.reserve(levels);
    pyramid.push_back(img);

    const Eigen::RowVectorXd KERNEL_1D = (Eigen::RowVectorXd(5) << 1.0, 4.0, 6.0, 4.0, 1.0).finished() / 16.0;

    for (int level_idx = 1; level_idx < levels; ++level_idx) {
        const Eigen::MatrixXd& PREV_LVL = pyramid.back();
        const int ROWS_COUNT = static_cast<int>(PREV_LVL.rows());
        const int COLS_COUNT = static_cast<int>(PREV_LVL.cols());

        if (ROWS_COUNT < 5 || COLS_COUNT < 5) {
            break;
        }

        // Gaussian Smoothing (Separable Convolution)
        Eigen::MatrixXd temp_blurred(ROWS_COUNT, COLS_COUNT);
        for (int r_idx = 0; r_idx < ROWS_COUNT; ++r_idx) {
            for (int c_idx = 0; c_idx < COLS_COUNT; ++c_idx) {
                double blur_sum = 0.0;
                for (int k_idx = -2; k_idx <= 2; ++k_idx) {
                    const int SAMPLED_COL = std::clamp(c_idx + k_idx, 0, COLS_COUNT - 1);
                    blur_sum += PREV_LVL(r_idx, SAMPLED_COL) * KERNEL_1D(k_idx + 2);
                }
                temp_blurred(r_idx, c_idx) = blur_sum;
            }
        }

        Eigen::MatrixXd fully_blurred(ROWS_COUNT, COLS_COUNT);
        for (int c_idx = 0; c_idx < COLS_COUNT; ++c_idx) {
            for (int r_idx = 0; r_idx < ROWS_COUNT; ++r_idx) {
                double blur_sum = 0.0;
                for (int k_idx = -2; k_idx <= 2; ++k_idx) {
                    const int SAMPLED_ROW = std::clamp(r_idx + k_idx, 0, ROWS_COUNT - 1);
                    blur_sum += temp_blurred(SAMPLED_ROW, c_idx) * KERNEL_1D(k_idx + 2);
                }
                fully_blurred(r_idx, c_idx) = blur_sum;
            }
        }

        // Downsampling
        const int NEW_ROWS = ROWS_COUNT / 2;
        const int NEW_COLS = COLS_COUNT / 2;
        if (NEW_ROWS < 3 || NEW_COLS < 3) {
            break;
        }

        Eigen::MatrixXd downsampled(NEW_ROWS, NEW_COLS);
        for (int r_idx = 0; r_idx < NEW_ROWS; ++r_idx) {
            for (int c_idx = 0; c_idx < NEW_COLS; ++c_idx) {
                downsampled(r_idx, c_idx) = fully_blurred(r_idx * 2, c_idx * 2);
            }
        }
        pyramid.push_back(downsampled);
    }
    return pyramid;
}

} // namespace vision

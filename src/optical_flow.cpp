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
#include <cmath>
#include <algorithm>

namespace vision {

static bool isOutOfBounds(const Eigen::Vector2d& pos, const Eigen::MatrixXd& img) {
    return (pos.x() < 1.0 || pos.y() < 1.0 || pos.x() >= static_cast<double>(img.cols()) - 1.0 || pos.y() >= static_cast<double>(img.rows()) - 1.0);
}

static bool fillLKDesignMatrix(
    const Eigen::MatrixXd& img_prev, const Eigen::MatrixXd& img_next,
    const Eigen::Vector2d& center,
    const Eigen::Vector2d& flow,
    int neighborhood_size,
    Eigen::MatrixXd& design_matrix, Eigen::VectorXd& observation_vector
) {
    const int HALF_WIN = neighborhood_size / 2;
    for(int row_offset = -HALF_WIN; row_offset <= HALF_WIN; ++row_offset) {
        for(int col_offset = -HALF_WIN; col_offset <= HALF_WIN; ++col_offset) {
            const int IDX_VAL = (row_offset + HALF_WIN) * neighborhood_size + (col_offset + HALF_WIN);
            const Eigen::Vector2d P_POS(center.x() + static_cast<double>(col_offset), center.y() + static_cast<double>(row_offset));
            const Eigen::Vector2d N_POS(P_POS.x() + flow.x(), P_POS.y() + flow.y());

            if (isOutOfBounds(P_POS, img_prev) || isOutOfBounds(N_POS, img_next)) {
                return false;
            }

            double grad_x = 0.0;
            double grad_y = 0.0;
            double grad_t = 0.0;
            computePixelGradients(img_prev, img_next, P_POS, N_POS, grad_x, grad_y, grad_t);

            design_matrix(IDX_VAL, 0) = grad_x;
            design_matrix(IDX_VAL, 1) = grad_y;
            observation_vector(IDX_VAL) = -grad_t;
        }
    }
    return true;
}

static int solveLKIteration(
    const Eigen::MatrixXd& img_prev,
    const Eigen::MatrixXd& img_next,
    double center_x, 
    double center_y,
    double& flow_dx,
    double& flow_dy,
    int neighborhood_size
) {
    const int NUM_ELEMENTS = neighborhood_size * neighborhood_size;
    Eigen::MatrixXd design_matrix(NUM_ELEMENTS, 2);
    Eigen::VectorXd observation_vector(NUM_ELEMENTS);
    
    const Eigen::Vector2d CENTER(center_x, center_y);
    const Eigen::Vector2d FLOW(flow_dx, flow_dy);

    if (!fillLKDesignMatrix(img_prev, img_next, CENTER, FLOW, neighborhood_size, design_matrix, observation_vector)) {
        return -1;
    }

    Eigen::Vector2d delta_flow(0.0, 0.0);
    const int IRLS_ITERATIONS = 3;
    const double HUBER_K = 0.1;

    for (int irls_idx = 0; irls_idx < IRLS_ITERATIONS; ++irls_idx) {
        Eigen::VectorXd weights(NUM_ELEMENTS);
        for (int i_idx = 0; i_idx < NUM_ELEMENTS; ++i_idx) {
            const double CURRENT_RESIDUAL = std::abs(design_matrix(i_idx, 0) * delta_flow.x() + design_matrix(i_idx, 1) * delta_flow.y() - observation_vector(i_idx));
            weights(i_idx) = (CURRENT_RESIDUAL <= HUBER_K) ? 1.0 : HUBER_K / CURRENT_RESIDUAL;
        }

        const Eigen::DiagonalMatrix<double, Eigen::Dynamic> WEIGHT_MAT(weights);
        const Eigen::Matrix2d CURRENT_HESSIAN = design_matrix.transpose() * WEIGHT_MAT * design_matrix;
        
        if (std::abs(CURRENT_HESSIAN.determinant()) < 1e-9) {
            return -1;
        }

        delta_flow = CURRENT_HESSIAN.ldlt().solve(design_matrix.transpose() * WEIGHT_MAT * observation_vector);
    }

    flow_dx += delta_flow.x();
    flow_dy += delta_flow.y();
    
    return (delta_flow.norm() < 0.001) ? 0 : 1;
}

void computePixelGradients(
    const Eigen::MatrixXd& img_prev, const Eigen::MatrixXd& img_next,
    const Eigen::Vector2d& prev_pos, const Eigen::Vector2d& next_pos,
    double& grad_x, double& grad_y, double& grad_t
) {
    const double POS_X_PREV = prev_pos.x();
    const double POS_Y_PREV = prev_pos.y();
    const double POS_X_NEXT = next_pos.x();
    const double POS_Y_NEXT = next_pos.y();

    const double VAL_I1_X = (bilinearInterpolation(img_prev, POS_X_PREV + 1.0, POS_Y_PREV) - bilinearInterpolation(img_prev, POS_X_PREV - 1.0, POS_Y_PREV)) * 0.5;
    const double VAL_I1_Y = (bilinearInterpolation(img_prev, POS_X_PREV, POS_Y_PREV + 1.0) - bilinearInterpolation(img_prev, POS_X_PREV, POS_Y_PREV - 1.0)) * 0.5;
    const double VAL_I2_X = (bilinearInterpolation(img_next, POS_X_NEXT + 1.0, POS_Y_NEXT) - bilinearInterpolation(img_next, POS_X_NEXT - 1.0, POS_Y_NEXT)) * 0.5;
    const double VAL_I2_Y = (bilinearInterpolation(img_next, POS_X_NEXT, POS_Y_NEXT + 1.0) - bilinearInterpolation(img_next, POS_X_NEXT, POS_Y_NEXT - 1.0)) * 0.5;
    
    grad_x = (VAL_I1_X + VAL_I2_X) * 0.5;
    grad_y = (VAL_I1_Y + VAL_I2_Y) * 0.5;
    grad_t = bilinearInterpolation(img_next, POS_X_NEXT, POS_Y_NEXT) - bilinearInterpolation(img_prev, POS_X_PREV, POS_Y_PREV);
}

static void trackFeatureAtLevel(
    const std::vector<Eigen::MatrixXd>& pyr_prev, const std::vector<Eigen::MatrixXd>& pyr_next,
    TrackedFeature& feat, double& flow_dx, double& flow_dy, 
    int neighborhood_size, bool& tracking_failed
) {
    const int MAX_ITERATIONS = 10;
    const int ACTUAL_LEVELS = static_cast<int>(pyr_prev.size());

    for (int level_idx = ACTUAL_LEVELS - 1; level_idx >= 0; --level_idx) {
        const double CURRENT_SCALE = std::pow(2.0, level_idx);
        const double POS_X_SCALED = feat.previous_pos.x() / CURRENT_SCALE;
        const double POS_Y_SCALED = feat.previous_pos.y() / CURRENT_SCALE;

        for (int iter_idx = 0; iter_idx < MAX_ITERATIONS; ++iter_idx) {
            const int CURRENT_RES_STATUS = solveLKIteration(pyr_prev[level_idx], pyr_next[level_idx], POS_X_SCALED, POS_Y_SCALED, flow_dx, flow_dy, neighborhood_size);
            if (CURRENT_RES_STATUS == 0) {
                break; 
            }
            if (CURRENT_RES_STATUS == -1) {
                tracking_failed = true;
                break; 
            }
        }

        if (tracking_failed) {
            break;
        }

        if (level_idx > 0) {
            flow_dx *= 2.0;
            flow_dy *= 2.0;
        }
    }
}

void calcOpticalFlowLK(
    const Eigen::MatrixXd& img_prev,
    const Eigen::MatrixXd& img_next,
    std::vector<TrackedFeature>& features,
    int neighborhood_size,
    int num_levels
) {
    Eigen::MatrixXd norm_prev = img_prev;
    Eigen::MatrixXd norm_next = img_next;
    
    applyLocalNormalization(norm_prev);
    applyLocalNormalization(norm_next);

    const std::vector<Eigen::MatrixXd> PYR_PREV = buildGaussianPyramid(norm_prev, num_levels);
    const std::vector<Eigen::MatrixXd> PYR_NEXT = buildGaussianPyramid(norm_next, num_levels);

    for (auto& feat : features) {
        if (feat.is_lost) {
            continue;
        }

        double flow_dx = 0.0;
        double flow_dy = 0.0;
        bool tracking_failed = false;

        trackFeatureAtLevel(PYR_PREV, PYR_NEXT, feat, flow_dx, flow_dy, neighborhood_size, tracking_failed);

        const double FINAL_POS_X = feat.previous_pos.x() + flow_dx;
        const double FINAL_POS_Y = feat.previous_pos.y() + flow_dy;
        
        if (FINAL_POS_X < 0.0 || FINAL_POS_Y < 0.0 || FINAL_POS_X >= static_cast<double>(img_prev.cols()) - 1.0 || FINAL_POS_Y >= static_cast<double>(img_prev.rows()) - 1.0) {
            tracking_failed = true;
        }

        if (!tracking_failed) {
            feat.current_pos = Eigen::Vector2d(FINAL_POS_X, FINAL_POS_Y);
        } else {
            feat.is_lost = true;
        }
    }
}

} // namespace vision
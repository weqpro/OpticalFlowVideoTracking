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

static void fillLKDesignMatrix(
    const Eigen::MatrixXd& img_prev, const Eigen::MatrixXd& img_next,
    double center_x, double center_y,
    double flow_dx, double flow_dy,
    int half_win, int neighborhood_size,
    Eigen::MatrixXd& design_matrix, Eigen::VectorXd& observation_vector, bool& out_of_bounds
) {
    for(int row_offset = -half_win; row_offset <= half_win; ++row_offset) {
        for(int col_offset = -half_win; col_offset <= half_win; ++col_offset) {
            const int idx_val = (row_offset + half_win) * neighborhood_size + (col_offset + half_win);
            const Eigen::Vector2d p_pos(center_x + static_cast<double>(col_offset), center_y + static_cast<double>(row_offset));
            const Eigen::Vector2d n_pos(p_pos.x() + flow_dx, p_pos.y() + flow_dy);

            if (isOutOfBounds(p_pos, img_prev) || isOutOfBounds(n_pos, img_next)) {
                out_of_bounds = true;
                return;
            }

            double grad_x = 0.0;
            double grad_y = 0.0;
            double grad_t = 0.0;
            computePixelGradients(img_prev, img_next, p_pos, n_pos, grad_x, grad_y, grad_t);

            design_matrix(idx_val, 0) = grad_x;
            design_matrix(idx_val, 1) = grad_y;
            observation_vector(idx_val) = -grad_t;
        }
    }
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
    const int HALF_WIN = neighborhood_size / 2;
    const int NUM_ELEMENTS = neighborhood_size * neighborhood_size;
    Eigen::MatrixXd design_matrix(NUM_ELEMENTS, 2);
    Eigen::VectorXd observation_vector(NUM_ELEMENTS);
    
    bool out_of_bounds = false;
    fillLKDesignMatrix(img_prev, img_next, center_x, center_y, flow_dx, flow_dy, HALF_WIN, neighborhood_size, design_matrix, observation_vector, out_of_bounds);
    if (out_of_bounds) {
        return -1;
    }

    Eigen::Vector2d delta_flow(0.0, 0.0);
    const int IRLS_ITERATIONS = 3;
    const double HUBER_K = 0.1;

    for (int irls_idx = 0; irls_idx < IRLS_ITERATIONS; ++irls_idx) {
        Eigen::VectorXd weights(NUM_ELEMENTS);
        for (int i_idx = 0; i_idx < NUM_ELEMENTS; ++i_idx) {
            const double current_residual = std::abs(design_matrix(i_idx, 0) * delta_flow.x() + design_matrix(i_idx, 1) * delta_flow.y() - observation_vector(i_idx));
            weights(i_idx) = (current_residual <= HUBER_K) ? 1.0 : HUBER_K / current_residual;
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
    const double pos_x_prev = prev_pos.x();
    const double pos_y_prev = prev_pos.y();
    const double pos_x_next = next_pos.x();
    const double pos_y_next = next_pos.y();

    const double val_i1_x = (bilinearInterpolation(img_prev, pos_x_prev + 1.0, pos_y_prev) - bilinearInterpolation(img_prev, pos_x_prev - 1.0, pos_y_prev)) * 0.5;
    const double val_i1_y = (bilinearInterpolation(img_prev, pos_x_prev, pos_y_prev + 1.0) - bilinearInterpolation(img_prev, pos_x_prev, pos_y_prev - 1.0)) * 0.5;
    const double val_i2_x = (bilinearInterpolation(img_next, pos_x_next + 1.0, pos_y_next) - bilinearInterpolation(img_next, pos_x_next - 1.0, pos_y_next)) * 0.5;
    const double val_i2_y = (bilinearInterpolation(img_next, pos_x_next, pos_y_next + 1.0) - bilinearInterpolation(img_next, pos_x_next, pos_y_next - 1.0)) * 0.5;
    
    grad_x = (val_i1_x + val_i2_x) * 0.5;
    grad_y = (val_i1_y + val_i2_y) * 0.5;
    grad_t = bilinearInterpolation(img_next, pos_x_next, pos_y_next) - bilinearInterpolation(img_prev, pos_x_prev, pos_y_prev);
}

static void trackFeatureAtLevel(
    const std::vector<Eigen::MatrixXd>& pyr_prev, const std::vector<Eigen::MatrixXd>& pyr_next,
    TrackedFeature& feat, double& flow_dx, double& flow_dy, 
    int neighborhood_size, bool& tracking_failed
) {
    const int MAX_ITERATIONS = 10;
    const int ACTUAL_LEVELS = static_cast<int>(pyr_prev.size());

    for (int level_idx = ACTUAL_LEVELS - 1; level_idx >= 0; --level_idx) {
        const double current_scale = std::pow(2.0, level_idx);
        const double pos_x_scaled = feat.previous_pos.x() / current_scale;
        const double pos_y_scaled = feat.previous_pos.y() / current_scale;

        for (int iter_idx = 0; iter_idx < MAX_ITERATIONS; ++iter_idx) {
            const int current_res_status = solveLKIteration(pyr_prev[level_idx], pyr_next[level_idx], pos_x_scaled, pos_y_scaled, flow_dx, flow_dy, neighborhood_size);
            if (current_res_status == 0) {
                break; 
            }
            if (current_res_status == -1) {
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

    const std::vector<Eigen::MatrixXd> pyr_prev = buildGaussianPyramid(norm_prev, num_levels);
    const std::vector<Eigen::MatrixXd> pyr_next = buildGaussianPyramid(norm_next, num_levels);

    for (auto& feat : features) {
        if (feat.is_lost) {
            continue;
        }

        double flow_dx = 0.0;
        double flow_dy = 0.0;
        bool tracking_failed = false;

        trackFeatureAtLevel(pyr_prev, pyr_next, feat, flow_dx, flow_dy, neighborhood_size, tracking_failed);

        const double final_pos_x = feat.previous_pos.x() + flow_dx;
        const double final_pos_y = feat.previous_pos.y() + flow_dy;
        
        if (final_pos_x < 0.0 || final_pos_y < 0.0 || final_pos_x >= static_cast<double>(img_prev.cols()) - 1.0 || final_pos_y >= static_cast<double>(img_prev.rows()) - 1.0) {
            tracking_failed = true;
        }

        if (!tracking_failed) {
            feat.current_pos = Eigen::Vector2d(final_pos_x, final_pos_y);
        } else {
            feat.is_lost = true;
        }
    }
}

} // namespace vision

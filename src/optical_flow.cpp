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

static void fillLKDesignMatrix(
    const Eigen::MatrixXd& img_prev, const Eigen::MatrixXd& img_next,
    const double CENTER_X, const double CENTER_Y,
    const double FLOW_DX, const double FLOW_DY,
    const int HALF_WIN, const int NEIGHBORHOOD_SIZE,
    Eigen::MatrixXd& design_matrix, Eigen::VectorXd& observation_vector, bool& out_of_bounds
) {
    for(int row_offset = -HALF_WIN; row_offset <= HALF_WIN; ++row_offset) {
        for(int col_offset = -HALF_WIN; col_offset <= HALF_WIN; ++col_offset) {
            const int IDX_VAL = (row_offset + HALF_WIN) * NEIGHBORHOOD_SIZE + (col_offset + HALF_WIN);
            const Eigen::Vector2d P_POS(CENTER_X + static_cast<double>(col_offset), CENTER_Y + static_cast<double>(row_offset));
            const Eigen::Vector2d N_POS(P_POS.x() + FLOW_DX, P_POS.y() + FLOW_DY);

            if (P_POS.x() < 1.0 || P_POS.y() < 1.0 || P_POS.x() >= static_cast<double>(img_prev.cols()) - 1.0 || P_POS.y() >= static_cast<double>(img_prev.rows()) - 1.0 ||
                N_POS.x() < 1.0 || N_POS.y() < 1.0 || N_POS.x() >= static_cast<double>(img_next.cols()) - 1.0 || N_POS.y() >= static_cast<double>(img_next.rows()) - 1.0) {
                out_of_bounds = true;
                return;
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
}

static int solveLKIteration(
    const Eigen::MatrixXd& img_prev,
    const Eigen::MatrixXd& img_next,
    const double CENTER_X, 
    const double CENTER_Y,
    double& flow_dx,
    double& flow_dy,
    const int NEIGHBORHOOD_SIZE
) {
    const int HALF_WIN = NEIGHBORHOOD_SIZE / 2;
    const int NUM_ELEMENTS = NEIGHBORHOOD_SIZE * NEIGHBORHOOD_SIZE;
    Eigen::MatrixXd design_matrix(NUM_ELEMENTS, 2);
    Eigen::VectorXd observation_vector(NUM_ELEMENTS);
    
    bool out_of_bounds = false;
    fillLKDesignMatrix(img_prev, img_next, CENTER_X, CENTER_Y, flow_dx, flow_dy, HALF_WIN, NEIGHBORHOOD_SIZE, design_matrix, observation_vector, out_of_bounds);
    if (out_of_bounds) {
        return -1;
    }

    Eigen::Vector2d delta_flow(0.0, 0.0);
    const int IRLS_ITERATIONS = 3;
    const double HUBER_K = 0.1;

    for (int irls_idx = 0; irls_idx < IRLS_ITERATIONS; ++irls_idx) {
        Eigen::VectorXd weights(NUM_ELEMENTS);
        for (int i_idx = 0; i_idx < NUM_ELEMENTS; ++i_idx) {
            const double RESIDUAL = std::abs(design_matrix(i_idx, 0) * delta_flow.x() + design_matrix(i_idx, 1) * delta_flow.y() - observation_vector(i_idx));
            weights(i_idx) = (RESIDUAL <= HUBER_K) ? 1.0 : HUBER_K / RESIDUAL;
        }

        const Eigen::DiagonalMatrix<double, Eigen::Dynamic> WEIGHT_MAT(weights);
        const Eigen::Matrix2d HESSIAN = design_matrix.transpose() * WEIGHT_MAT * design_matrix;
        
        if (std::abs(HESSIAN.determinant()) < 1e-9) {
            return -1;
        }

        delta_flow = HESSIAN.ldlt().solve(design_matrix.transpose() * WEIGHT_MAT * observation_vector);
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

static void trackFeatureAtLevel(
    const std::vector<Eigen::MatrixXd>& PYR_PREV, const std::vector<Eigen::MatrixXd>& PYR_NEXT,
    TrackedFeature& feat, double& flow_dx, double& flow_dy, 
    const int NEIGHBORHOOD_SIZE, bool& tracking_failed
) {
    const int MAX_ITERATIONS = 10;
    const int ACTUAL_LEVELS = static_cast<int>(PYR_PREV.size());

    for (int level_idx = ACTUAL_LEVELS - 1; level_idx >= 0; --level_idx) {
        const double SCALE = std::pow(2.0, level_idx);
        const double POS_X = feat.previous_pos.x() / SCALE;
        const double POS_Y = feat.previous_pos.y() / SCALE;

        for (int iter_idx = 0; iter_idx < MAX_ITERATIONS; ++iter_idx) {
            const int RES_STATUS = solveLKIteration(PYR_PREV[level_idx], PYR_NEXT[level_idx], POS_X, POS_Y, flow_dx, flow_dy, NEIGHBORHOOD_SIZE);
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
    const int NEIGHBORHOOD_SIZE,
    const int NUM_LEVELS
) {
    Eigen::MatrixXd norm_prev = img_prev;
    Eigen::MatrixXd norm_next = img_next;
    
    applyLocalNormalization(norm_prev);
    applyLocalNormalization(norm_next);

    const std::vector<Eigen::MatrixXd> PYR_PREV = buildGaussianPyramid(norm_prev, NUM_LEVELS);
    const std::vector<Eigen::MatrixXd> PYR_NEXT = buildGaussianPyramid(norm_next, NUM_LEVELS);

    for (auto& feat : features) {
        if (feat.is_lost) {
            continue;
        }

        double flow_dx = 0.0;
        double flow_dy = 0.0;
        bool tracking_failed = false;

        trackFeatureAtLevel(PYR_PREV, PYR_NEXT, feat, flow_dx, flow_dy, NEIGHBORHOOD_SIZE, tracking_failed);

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

} // namespace vision

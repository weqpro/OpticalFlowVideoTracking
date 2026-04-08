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

double bilinearInterpolation(const Eigen::MatrixXd& mat, double x_coord, double y_coord) {
    int x_base = static_cast<int>(std::floor(x_coord));
    int y_base = static_cast<int>(std::floor(y_coord));

    if (x_base < 0 || y_base < 0 || x_base + 1 >= mat.cols() || y_base + 1 >= mat.rows()) {
        return mat(std::clamp(y_base, 0, static_cast<int>(mat.rows() - 1)), 
                   std::clamp(x_base, 0, static_cast<int>(mat.cols() - 1)));
    }

    double delta_x = x_coord - x_base;
    double delta_y = y_coord - y_base;
    
    double val00 = mat(y_base, x_base);         
    double val10 = mat(y_base, x_base + 1);     
    double val01 = mat(y_base + 1, x_base);     
    double val11 = mat(y_base + 1, x_base + 1); 
    
    return (val00 * (1.0 - delta_x) * (1.0 - delta_y) +
            val10 * delta_x * (1.0 - delta_y) +
            val01 * (1.0 - delta_x) * delta_y +
            val11 * delta_x * delta_y);
}

void computeSpatialGradients(const Eigen::MatrixXd& image, Eigen::MatrixXd& Ix, Eigen::MatrixXd& Iy) {
    const int rows_count = static_cast<int>(image.rows());
    const int cols_count = static_cast<int>(image.cols());
    Ix.setZero(rows_count, cols_count);
    Iy.setZero(rows_count, cols_count);
    
    Eigen::Matrix3d kernel_x;
    Eigen::Matrix3d kernel_y;
    kernel_x << -3, 0, 3, -10, 0, 10, -3, 0, 3;
    kernel_y << -3, -10, -3, 0, 0, 0, 3, 10, 3;

    for (int i = 1; i < rows_count - 1; ++i) {
        for (int j = 1; j < cols_count - 1; ++j) {
            auto region = image.block<3, 3>(i - 1, j - 1).array();
            Ix(i, j) = (region * kernel_x.array()).sum();
            Iy(i, j) = (region * kernel_y.array()).sum();
        }
    }
}

Eigen::MatrixXd computeMinEigenvalueMap(const Eigen::MatrixXd& Ix, const Eigen::MatrixXd& Iy) {
    const int rows_count = static_cast<int>(Ix.rows());
    const int cols_count = static_cast<int>(Ix.cols());
    
    Eigen::MatrixXd Ixx = Ix.array().square();
    Eigen::MatrixXd Iyy = Iy.array().square();
    Eigen::MatrixXd Ixy = Ix.array() * Iy.array();
    Eigen::MatrixXd eig_min = Eigen::MatrixXd::Zero(rows_count, cols_count);

    for (int i = 1; i < rows_count - 1; ++i) {
        for (int j = 1; j < cols_count - 1; ++j) {
            double sum_xx = Ixx.block<3, 3>(i - 1, j - 1).sum();
            double sum_yy = Iyy.block<3, 3>(i - 1, j - 1).sum();
            double sum_xy = Ixy.block<3, 3>(i - 1, j - 1).sum();
            eig_min(i, j) = 0.5 * (sum_xx + sum_yy - std::sqrt((sum_xx - sum_yy) * (sum_xx - sum_yy) + 4.0 * sum_xy * sum_xy));
        }
    }
    return eig_min;
}

std::vector<CornerCandidate> collectLocalMaxima(const Eigen::MatrixXd& eig_min, double threshold) {
    const int rows_count = static_cast<int>(eig_min.rows());
    const int cols_count = static_cast<int>(eig_min.cols());
    std::vector<CornerCandidate> candidates;

    for (int i = 1; i < rows_count - 1; ++i) {
        for (int j = 1; j < cols_count - 1; ++j) {
            double val = eig_min(i, j);
            if (val > threshold) {
                bool is_local_max = true;
                for (int ni = -1; ni <= 1; ++ni) {
                    for (int nj = -1; nj <= 1; ++nj) {
                        if (ni == 0 && nj == 0) continue;
                        if (eig_min(i + ni, j + nj) > val) {
                            is_local_max = false;
                            break;
                        }
                    }
                    if (!is_local_max) break;
                }
                if (is_local_max) candidates.push_back({i, j, val});
            }
        }
    }
    return candidates;
}

// --- Public API Implementation ---

std::vector<Eigen::Vector2d> findGoodFeaturesToTrack(
    const Eigen::MatrixXd& image,
    int max_corners,
    double quality_level,
    double min_distance
) {
    if (image.rows() < 3 || image.cols() < 3) return {};

    Eigen::MatrixXd Ix;
    Eigen::MatrixXd Iy;
    computeSpatialGradients(image, Ix, Iy);
    
    Eigen::MatrixXd eig_min = computeMinEigenvalueMap(Ix, Iy);
    
    double threshold = eig_min.maxCoeff() * quality_level;
    auto candidates = collectLocalMaxima(eig_min, threshold);

    std::sort(candidates.begin(), candidates.end(), [](const CornerCandidate& lhs, const CornerCandidate& rhs) {
        return lhs.val > rhs.val;
    });

    std::vector<Eigen::Vector2d> corners;
    for (const auto& cand : candidates) {
        if (corners.size() >= static_cast<size_t>(max_corners)) break;
        
        bool far_enough = true;
        for (const auto& existing : corners) {
            if ((Eigen::Vector2d(cand.c, cand.r) - existing).norm() < min_distance) {
                far_enough = false;
                break;
            }
        }
        if (far_enough) corners.emplace_back(cand.c, cand.r);
    }
    return corners;
}

static bool solveLKSystem(
    const Eigen::MatrixXd& grad_x, 
    const Eigen::MatrixXd& grad_y, 
    const Eigen::MatrixXd& grad_t,
    double center_x, 
    double center_y, 
    int neighborhood_size,
    Eigen::Vector2d& flow_vector
) {
    int half_win = neighborhood_size / 2;
    const int num_elements = neighborhood_size * neighborhood_size;
    Eigen::MatrixXd design_matrix(num_elements, 2);
    Eigen::VectorXd observation_vector(num_elements);

    for(int i = 0; i < neighborhood_size; ++i) {
        for(int j = 0; j < neighborhood_size; ++j) {
            int idx = i * neighborhood_size + j;
            double cur_x = center_x - half_win + i;
            double cur_y = center_y - half_win + j;

            design_matrix(idx, 0) = bilinearInterpolation(grad_x, cur_x, cur_y);
            design_matrix(idx, 1) = bilinearInterpolation(grad_y, cur_x, cur_y);
            observation_vector(idx) = -bilinearInterpolation(grad_t, cur_x, cur_y);
        }
    }

    Eigen::Matrix2d hessian = design_matrix.transpose() * design_matrix;
    if (std::abs(hessian.determinant()) < 1e-6) {
        return false;
    }

    flow_vector = hessian.ldlt().solve(design_matrix.transpose() * observation_vector);
    return true;
}

void calcOpticalFlowLK(
    const Eigen::MatrixXd& img_prev,
    const Eigen::MatrixXd& img_next,
    std::vector<TrackedFeature>& features,
    int neighborhood_size
) {
    Eigen::MatrixXd grad_x;
    Eigen::MatrixXd grad_y;
    Eigen::MatrixXd grad_t;
    vision::computeDerivatives(img_prev, img_next, grad_x, grad_y, grad_t);
    int half_win = neighborhood_size / 2;

    for (auto& feat : features) {
        if (feat.is_lost) continue;

        double pos_x = feat.previous_pos.x();
        double pos_y = feat.previous_pos.y();

        if (pos_x - half_win < 0.0 || pos_y - half_win < 0.0 || 
            pos_x + half_win + 1.0 >= static_cast<double>(img_prev.cols()) || 
            pos_y + half_win + 1.0 >= static_cast<double>(img_prev.rows())) {
            feat.is_lost = true;
            continue;
        }

        Eigen::Vector2d flow_vector;
        if (solveLKSystem(grad_x, grad_y, grad_t, pos_x, pos_y, neighborhood_size, flow_vector)) {
            feat.current_pos = feat.previous_pos + flow_vector;
        } else {
            feat.is_lost = true;
        }
    }
}

std::vector<Eigen::MatrixXd> buildGaussianPyramid(const Eigen::MatrixXd& img, int levels) {
    std::vector<Eigen::MatrixXd> pyramid;
    pyramid.push_back(img);

    for (int i = 1; i < levels; ++i) {
        const auto& prev_img = pyramid.back();
        int new_rows = static_cast<int>(prev_img.rows() / 2);
        int new_cols = static_cast<int>(prev_img.cols() / 2);
        
        if (new_rows < 3 || new_cols < 3) break;

        Eigen::MatrixXd downsampled(new_rows, new_cols);
        for (int r = 0; r < new_rows; ++r) {
            for (int c = 0; c < new_cols; ++c) {
                downsampled(r, c) = prev_img.block<2, 2>(r * 2, c * 2).mean();
            }
        }
        pyramid.push_back(downsampled);
    }
    return pyramid;
}

} // namespace vision

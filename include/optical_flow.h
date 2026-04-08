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

#ifndef OPTICAL_FLOW_H
#define OPTICAL_FLOW_H
#include <Eigen/Dense> 
#include <vector>

namespace vision {

struct CornerCandidate { 
    int r, c; 
    double val; 
};

void computeSpatialGradients(
    const Eigen::MatrixXd& image, 
    Eigen::MatrixXd& grad_ix, 
    Eigen::MatrixXd& grad_iy
);

Eigen::MatrixXd computeMinEigenvalueMap(
    const Eigen::MatrixXd& grad_ix, 
    const Eigen::MatrixXd& grad_iy
);

std::vector<CornerCandidate> collectLocalMaxima(
    const Eigen::MatrixXd& eig_min, 
    double threshold
);

std::vector<Eigen::Vector2d> findGoodFeaturesToTrack(
    const Eigen::MatrixXd& image,
    int max_corners = 100,
    double quality_level = 0.01,
    double min_distance = 10.0
);

struct TrackedFeature {
    Eigen::Vector2d previous_pos{Eigen::Vector2d::Zero()}; 
    Eigen::Vector2d current_pos{Eigen::Vector2d::Zero()}; 
    bool is_lost{false}; 

    TrackedFeature(Eigen::Vector2d previous, Eigen::Vector2d current, bool lost = false)
    : previous_pos(std::move(previous)), current_pos(std::move(current)), is_lost(lost) {}
};

void calcOpticalFlowLK(
    const Eigen::MatrixXd& img_prev,
    const Eigen::MatrixXd& img_next,
    std::vector<TrackedFeature>& features,
    int neighborhood_size = 3
);

double bilinearInterpolation(
    const Eigen::MatrixXd& mat,
    double x_coord,
    double y_coord);

std::vector<Eigen::MatrixXd> buildGaussianPyramid(
    const Eigen::MatrixXd& img, 
    int levels
);

} // namespace vision

#endif // OPTICAL_FLOW_H
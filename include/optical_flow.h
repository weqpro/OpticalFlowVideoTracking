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
#include "vision_types.h"
#include "image_utils.h"
#include "feature_detector.h"

namespace vision {

void calcOpticalFlowLK(
    const Eigen::MatrixXd& img_prev,
    const Eigen::MatrixXd& img_next,
    std::vector<TrackedFeature>& features,
    int neighborhood_size = 3,
    int num_levels = 1
);

void computePixelGradients(
    const Eigen::MatrixXd& img_prev, const Eigen::MatrixXd& img_next,
    const Eigen::Vector2d& prev_pos, const Eigen::Vector2d& next_pos,
    double& grad_x, double& grad_y, double& grad_t
);

} // namespace vision

#endif // OPTICAL_FLOW_H

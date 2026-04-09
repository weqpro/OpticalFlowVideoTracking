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

#ifndef FEATURE_DETECTOR_H
#define FEATURE_DETECTOR_H

#include <Eigen/Dense>
#include <vector>
#include "vision_types.h"

namespace vision {

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
    double THRESHOLD
);

std::vector<Eigen::Vector2d> findGoodFeaturesToTrack(
    const Eigen::MatrixXd& image,
    int MAX_CORNERS = 100,
    double QUALITY_LEVEL = 0.01,
    double MIN_DISTANCE = 10.0
);

} // namespace vision

#endif // FEATURE_DETECTOR_H

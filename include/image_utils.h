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

#ifndef IMAGE_UTILS_H
#define IMAGE_UTILS_H

#include <Eigen/Dense>
#include <vector>

namespace vision {

double bilinearInterpolation(const Eigen::MatrixXd& mat, double X_COORD, double Y_COORD);

std::vector<Eigen::MatrixXd> buildGaussianPyramid(const Eigen::MatrixXd& img, int LEVELS);

void applyLocalNormalization(Eigen::MatrixXd& img);

} // namespace vision

#endif // IMAGE_UTILS_H

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

#include "deriv.h"
#include "optical_flow.h"

namespace vision {

void computeDerivatives(
    const Eigen::MatrixXd& img_prev, 
    const Eigen::MatrixXd& img_next, 
    Eigen::MatrixXd& grad_x, 
    Eigen::MatrixXd& grad_y, 
    Eigen::MatrixXd& grad_t
) {
    const int rows = static_cast<int>(img_prev.rows());
    const int cols = static_cast<int>(img_prev.cols());

    grad_x.setZero(rows, cols);
    grad_y.setZero(rows, cols);
    grad_t = img_next - img_prev;

    for (int y = 1; y < rows - 1; ++y) {
        for (int x = 1; x < cols - 1; ++x) {
            grad_x(y, x) = (img_prev(y, x + 1) - img_prev(y, x - 1) + 
                            img_next(y, x + 1) - img_next(y, x - 1)) / 4.0;
            grad_y(y, x) = (img_prev(y + 1, x) - img_prev(y - 1, x) + 
                            img_next(y + 1, x) - img_next(y - 1, x)) / 4.0;
        }
    }
}

} // namespace vision

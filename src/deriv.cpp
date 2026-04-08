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
    const int ROWS = static_cast<int>(img_prev.rows());
    const int COLS = static_cast<int>(img_prev.cols());

    grad_x.setZero(ROWS, COLS);
    grad_y.setZero(ROWS, COLS);
    grad_t = img_next - img_prev;

    for (int row_idx = 1; row_idx < ROWS - 1; ++row_idx) {
        for (int col_idx = 1; col_idx < COLS - 1; ++col_idx) {
            grad_x(row_idx, col_idx) = (img_prev(row_idx, col_idx + 1) - img_prev(row_idx, col_idx - 1) + 
                            img_next(row_idx, col_idx + 1) - img_next(row_idx, col_idx - 1)) / 4.0;
            grad_y(row_idx, col_idx) = (img_prev(row_idx + 1, col_idx) - img_prev(row_idx - 1, col_idx) + 
                            img_next(row_idx + 1, col_idx) - img_next(row_idx - 1, col_idx)) / 4.0;
        }
    }
}

} // namespace vision

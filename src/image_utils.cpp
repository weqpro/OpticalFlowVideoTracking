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

#include "image_utils.h"
#include <cmath>
#include <algorithm>

namespace vision {

double bilinearInterpolation(const Eigen::MatrixXd& mat, const double X_COORD, const double Y_COORD) {
    const int X_BASE = static_cast<int>(std::floor(X_COORD));
    const int Y_BASE = static_cast<int>(std::floor(Y_COORD));

    if (X_BASE < 0 || Y_BASE < 0 || X_BASE + 1 >= mat.cols() || Y_BASE + 1 >= mat.rows()) {
        return mat(std::clamp(Y_BASE, 0, static_cast<int>(mat.rows() - 1)), 
                   std::clamp(X_BASE, 0, static_cast<int>(mat.cols() - 1)));
    }

    const double DELTA_X = X_COORD - static_cast<double>(X_BASE);
    const double DELTA_Y = Y_COORD - static_cast<double>(Y_BASE);
    
    const double VAL00 = mat(Y_BASE, X_BASE);         
    const double VAL10 = mat(Y_BASE, X_BASE + 1);     
    const double VAL01 = mat(Y_BASE + 1, X_BASE);     
    const double VAL11 = mat(Y_BASE + 1, X_BASE + 1); 
    
    return (VAL00 * (1.0 - DELTA_X) * (1.0 - DELTA_Y) +
            VAL10 * DELTA_X * (1.0 - DELTA_Y) +
            VAL01 * (1.0 - DELTA_X) * DELTA_Y +
            VAL11 * DELTA_X * DELTA_Y);
}

static Eigen::MatrixXd applyGaussianBlur(const Eigen::MatrixXd& input) {
    const int ROWS = static_cast<int>(input.rows());
    const int COLS = static_cast<int>(input.cols());
    const Eigen::RowVectorXd KERNEL_1D = (Eigen::RowVectorXd(5) << 1.0, 4.0, 6.0, 4.0, 1.0).finished() / 16.0;

    Eigen::MatrixXd temp_blurred(ROWS, COLS);
    for (int row_idx = 0; row_idx < ROWS; ++row_idx) {
        for (int col_idx = 0; col_idx < COLS; ++col_idx) {
            double blur_sum = 0.0;
            for (int k_idx = -2; k_idx <= 2; ++k_idx) {
                const int SAMPLED_COL = std::clamp(col_idx + k_idx, 0, COLS - 1);
                blur_sum += input(row_idx, SAMPLED_COL) * KERNEL_1D(k_idx + 2);
            }
            temp_blurred(row_idx, col_idx) = blur_sum;
        }
    }

    Eigen::MatrixXd fully_blurred(ROWS, COLS);
    for (int col_idx = 0; col_idx < COLS; ++col_idx) {
        for (int row_idx = 0; row_idx < ROWS; ++row_idx) {
            double blur_sum = 0.0;
            for (int k_idx = -2; k_idx <= 2; ++k_idx) {
                const int SAMPLED_ROW = std::clamp(row_idx + k_idx, 0, ROWS - 1);
                blur_sum += temp_blurred(SAMPLED_ROW, col_idx) * KERNEL_1D(k_idx + 2);
            }
            fully_blurred(row_idx, col_idx) = blur_sum;
        }
    }
    return fully_blurred;
}

std::vector<Eigen::MatrixXd> buildGaussianPyramid(const Eigen::MatrixXd& img, const int LEVELS) {
    std::vector<Eigen::MatrixXd> pyramid;
    pyramid.reserve(LEVELS);
    pyramid.push_back(img);

    for (int level_idx = 1; level_idx < LEVELS; ++level_idx) {
        const auto& prev_lvl = pyramid.back();
        const int ROWS = static_cast<int>(prev_lvl.rows());
        const int COLS = static_cast<int>(prev_lvl.cols());

        if (ROWS < 5 || COLS < 5) {
            break;
        }

        Eigen::MatrixXd fully_blurred = applyGaussianBlur(prev_lvl);

        const int NEW_ROWS = ROWS / 2;
        const int NEW_COLS = COLS / 2;
        if (NEW_ROWS < 3 || NEW_COLS < 3) {
            break;
        }

        Eigen::MatrixXd downsampled(NEW_ROWS, NEW_COLS);
        for (int row_idx = 0; row_idx < NEW_ROWS; ++row_idx) {
            for (int col_idx = 0; col_idx < NEW_COLS; ++col_idx) {
                downsampled(row_idx, col_idx) = fully_blurred(static_cast<Eigen::Index>(row_idx) * 2, static_cast<Eigen::Index>(col_idx) * 2);
            }
        }
        pyramid.push_back(downsampled);
    }
    return pyramid;
}

void applyLocalNormalization(Eigen::MatrixXd& img) {
    const int ROWS = static_cast<int>(img.rows());
    const int COLS = static_cast<int>(img.cols());
    const int WIN = 5; 
    const int HALF_WIN = WIN / 2;
    
    Eigen::MatrixXd temp = img;
    Eigen::MatrixXd mean_img = Eigen::MatrixXd::Zero(ROWS, COLS);

    for (int row_idx = 0; row_idx < ROWS; ++row_idx) {
        double row_sum = 0;
        for (int i_idx = -HALF_WIN; i_idx <= HALF_WIN; ++i_idx) {
            row_sum += img(row_idx, std::clamp(i_idx, 0, COLS - 1));
        }
        for (int col_idx = 0; col_idx < COLS; ++col_idx) {
            temp(row_idx, col_idx) = row_sum / WIN;
            row_sum += img(row_idx, std::clamp(col_idx + HALF_WIN + 1, 0, COLS - 1)) - img(row_idx, std::clamp(col_idx - HALF_WIN, 0, COLS - 1));
        }
    }

    for (int col_idx = 0; col_idx < COLS; ++col_idx) {
        double col_sum = 0;
        for (int i_idx = -HALF_WIN; i_idx <= HALF_WIN; ++i_idx) {
            col_sum += temp(std::clamp(i_idx, 0, ROWS - 1), col_idx);
        }
        for (int row_idx = 0; row_idx < ROWS; ++row_idx) {
            mean_img(row_idx, col_idx) = col_sum / WIN;
            col_sum += temp(std::clamp(row_idx + HALF_WIN + 1, 0, ROWS - 1), col_idx) - temp(std::clamp(row_idx - HALF_WIN, 0, ROWS - 1));
        }
    }
    
    img -= mean_img;
}

} // namespace vision

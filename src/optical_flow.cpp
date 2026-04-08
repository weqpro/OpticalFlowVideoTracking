#include "optical_flow.h"
#include <iostream>
#include <cmath>

double bilinearInterpolation(const Eigen::MatrixXd& mat, double x, double y) {
    int x0 = std::floor(x);
    int y0 = std::floor(y);

    double dx = x - x0;
    double dy = y - y0;
    
    double v00 = mat(y0, x0);         
    double v10 = mat(y0, x0 + 1);     
    double v01 = mat(y0 + 1, x0);     
    double v11 = mat(y0 + 1, x0 + 1); 
    
    return (v00 * (1 - dx) * (1 - dy) +
            v10 * dx * (1 - dy) +
            v01 * (1 - dx) * dy +
            v11 * dx * dy);
}

void calcOpticalFlowLK(
    const Eigen::MatrixXd& img_prev,
    const Eigen::MatrixXd& img_next,
    std::vector<TrackedFeature>& features,
    int neighborhood_size
) {
    Eigen::MatrixXd grad_x, grad_y, grad_t;
    vision::computeDerivatives(img_prev, img_next, grad_x, grad_y, grad_t);
    int half_win = neighborhood_size / 2;

    for (auto& feat : features) {
        if (feat.is_lost) { continue; }

        double x = feat.previous_pos.x();
        double y = feat.previous_pos.y();

        if (x - half_win < 0 || y - half_win < 0 || 
            x + half_win + 1 >= img_prev.cols() || y + half_win + 1 >= img_prev.rows()) {
            feat.is_lost = true;
            continue;
        }

        // construct the design_matrix and observation_vector for the least squares problem
        const unsigned int NUM_ELEMENTS = neighborhood_size * neighborhood_size;
        Eigen::MatrixXd design_matrix(NUM_ELEMENTS, 2);
        Eigen::MatrixXd observation_vector(NUM_ELEMENTS, 1);

        for(int i = 0; i < neighborhood_size; ++i) {
            for(int j = 0; j < neighborhood_size; ++j) {
                int idx = i * neighborhood_size + j;

                double cur_x = x - half_win + i;
                double cur_y = y - half_win + j;

                double gx = bilinearInterpolation(grad_x, cur_x, cur_y);
                double gy = bilinearInterpolation(grad_y, cur_x, cur_y);
                double gt = bilinearInterpolation(grad_t, cur_x, cur_y);

                design_matrix(idx, 0) = gx;
                design_matrix(idx, 1) = gy;
                observation_vector(idx, 0) = -gt;
            }
        }
        Eigen::Matrix2d hessian_matrix = design_matrix.transpose() * design_matrix;
        Eigen::Vector2d b_new = design_matrix.transpose() * observation_vector;

        // check if hessian_matrix is invertible
        if (std::abs(hessian_matrix.determinant()) < 1e-6) {
            feat.is_lost = true;
            continue;
        }

        Eigen::Vector2d flow_vector = hessian_matrix.inverse() * b_new;
        feat.current_pos = feat.previous_pos + flow_vector;
    }
}
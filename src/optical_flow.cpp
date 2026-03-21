#include "optical_flow.h"
#include <iostream>
#include <cmath>

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
        int pixel_x = std::round(feat.previous_pos.x()); 
        int pixel_y = std::round(feat.previous_pos.y());

        // check if the neighborhood is fully contained in the image
        if (((pixel_x-half_win < 0) || (pixel_y-half_win < 0)) 
         || ((pixel_x+half_win >= img_prev.cols()) || (pixel_y+half_win >= img_prev.rows()) )) {
            feat.is_lost = true;
            continue;
        }

        // construct the design_matrix and observation_vector for the least squares problem
        const unsigned int num_elements = neighborhood_size * neighborhood_size;
        Eigen::MatrixXd design_matrix(num_elements, 2);
        Eigen::MatrixXd observation_vector(num_elements, 1);

        for(int dx=0; dx<neighborhood_size; ++dx) {
            for(int dy=0; dy<neighborhood_size; ++dy) {
                int idx = dx * neighborhood_size + dy;
                design_matrix(idx, 0) = grad_x(pixel_y - half_win + dy, pixel_x - half_win + dx);
                design_matrix(idx, 1) = grad_y(pixel_y - half_win + dy, pixel_x - half_win + dx);
                observation_vector(idx, 0) = -grad_t(pixel_y - half_win + dy, pixel_x - half_win + dx);
            }
        }

        // solve for the flow vector using least squares
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
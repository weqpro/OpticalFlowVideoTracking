#include "optical_flow.h"
#include <iostream>
#include <Eigen/Dense>
#include <cmath>

void calcOpticalFlowLK(
    const Eigen::MatrixXd& img_prev,
    const Eigen::MatrixXd& img_next,
    std::vector<TrackedFeature>& features,
    int neighborhood_size
) {
    Eigen::MatrixXd Ix, Iy, It;
    vision::computeDerivatives(img_prev, img_next, Ix, Iy, It);
    int half_win = neighborhood_size / 2;

    for (auto& feat : features) {
        if (feat.is_lost) continue;
        int px = std::round(feat.previous_pos.x()); 
        int py = std::round(feat.previous_pos.y());

        // check if the neighborhood is fully contained in the image
        if (((px-half_win < 0) || (py-half_win < 0)) 
         || ((px+half_win >= img_prev.cols()) || (py+half_win >= img_prev.rows()) )) {
            feat.is_lost = true;
            continue;
        }

        // construct the A matrix and b vector for the least squares problem
        const unsigned int N = neighborhood_size * neighborhood_size;
        Eigen::MatrixXd A(N, 2);
        Eigen::MatrixXd b(N, 1);

        for(int dx=0; dx<neighborhood_size; ++dx) {
            for(int dy=0; dy<neighborhood_size; ++dy) {
                int idx = dx * neighborhood_size + dy;
                A(idx, 0) = Ix(py - half_win + dy, px - half_win + dx);
                A(idx, 1) = Iy(py - half_win + dy, px - half_win + dx);
                b(idx, 0) = -It(py - half_win + dy, px - half_win + dx);
            }
        }

        // solve for the flow vector using least squares
        Eigen::Matrix2d H = A.transpose() * A;
        Eigen::Vector2d b_new = A.transpose() * b;

        // check if H is invertible
        if (std::abs(H.determinant()) < 1e-6) {
            feat.is_lost = true;
            continue;
        }

        Eigen::Vector2d v = H.inverse() * b_new;
        feat.current_pos = feat.previous_pos + v;
    }
}
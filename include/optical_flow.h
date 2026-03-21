#ifndef OPTICAL_FLOW_H
#define OPTICAL_FLOW_H

namespace vision {

void computeDerivatives(
    const Eigen::MatrixXd& img_prev, 
    const Eigen::MatrixXd& img_next, 
    Eigen::MatrixXd& Ix, 
    Eigen::MatrixXd& Iy, 
    Eigen::MatrixXd& It
);

std::vector<Eigen::Vector2d> findGoodFeaturesToTrack(
    const Eigen::MatrixXd& image,
    int max_corners = 100,
    double quality_level = 0.01,
    double min_distance = 10.0
);

} // !namespace vision

struct TrackedFeature {
    Eigen::Vector2d previous_pos;
    Eigen::Vector2d current_pos;
    bool is_lost{false};
};

void calcOpticalFlowLK(
    const Eigen::MatrixXd& img_prev,
    const Eigen::MatrixXd& img_next,
    std::vector<TrackedFeature>& features,
    int neighborhood_size = 3
);

#endif // OPTICAL_FLOW_H
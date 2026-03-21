#ifndef OPTICAL_FLOW_H
#define OPTICAL_FLOW_H
#include <Eigen/Dense> 
#include <vector>

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
    Eigen::Vector2d previous_pos{Eigen::Vector2d::Zero()}; 
    Eigen::Vector2d current_pos{Eigen::Vector2d::Zero()}; 
    bool is_lost{false}; 
    
    TrackedFeature(const Eigen::Vector2d& previous, const Eigen::Vector2d& current, bool lost = false) : 
    previous_pos(previous), current_pos(current), is_lost(lost) 
    {}
};

void calcOpticalFlowLK(
    const Eigen::MatrixXd& img_prev,
    const Eigen::MatrixXd& img_next,
    std::vector<TrackedFeature>& features,
    int neighborhood_size = 3
);

#endif // OPTICAL_FLOW_H
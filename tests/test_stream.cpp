#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>

#include "optical_flow.h" 

// Створює чорне зображення з білим квадратом посередині
Eigen::MatrixXd createTestImage(int rows, int cols, int square_x, int square_y, int square_size) {
    Eigen::MatrixXd img = Eigen::MatrixXd::Zero(rows, cols);
    for (int y = square_y; y < square_y + square_size; ++y) {
        for (int x = square_x; x < square_x + square_size; ++x) {
            if (x >= 0 && x < cols && y >= 0 && y < rows) {
                img(y, x) = 1.0;
            }
        }
    }
    return img;
}
// 1. ТЕСТ НА ЗБІГ З OPENCV
TEST(OpticalFlowTest, CalcOpticalFlowLK_SimpleTranslation_MatchesOpenCV) {
    // 1. Arrange
    int rows = 50;
    int cols = 50;
    int win_size = 3; 
    
    Eigen::MatrixXd img_prev = createTestImage(rows, cols, 20, 20, 10);
    Eigen::MatrixXd img_next = createTestImage(rows, cols, 21, 21, 10);

    vision::TrackedFeature my_feat(Eigen::Vector2d(25.0, 25.0), Eigen::Vector2d(25.0, 25.0), false);
    std::vector<vision::TrackedFeature> my_features = {my_feat};

    cv::Mat cv_img_prev, cv_img_next;
    cv::eigen2cv(img_prev, cv_img_prev);
    cv::eigen2cv(img_next, cv_img_next);
    cv_img_prev.convertTo(cv_img_prev, CV_32F); 
    cv_img_next.convertTo(cv_img_next, CV_32F);

    std::vector<cv::Point2f> cv_prev_pts = { cv::Point2f(25.0f, 25.0f) };
    std::vector<cv::Point2f> cv_next_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    // 2. Act
    vision::calcOpticalFlowLK(img_prev, img_next, my_features, win_size);
    cv::calcOpticalFlowPyrLK(cv_img_prev, cv_img_next, cv_prev_pts, cv_next_pts, 
                             status, err, cv::Size(win_size, win_size), 0);

    // 3. Assert
    ASSERT_FALSE(my_features[0].is_lost) << "Точка не повинна бути втрачена";
    ASSERT_EQ(status[0], 1) << "OpenCV також повинен знайти точку";

    double tol = 0.5; 
    EXPECT_NEAR(my_features[0].current_pos.x(), cv_next_pts[0].x, tol);
    EXPECT_NEAR(my_features[0].current_pos.y(), cv_next_pts[0].y, tol);
}

// 2. ТЕСТ НА ВИХІД ЗА МЕЖІ (Boundary Check)
TEST(OpticalFlowTest, CalcOpticalFlowLK_FeatureNearEdge_MarksAsLost) {
    // 1. Arrange
    Eigen::MatrixXd img_prev = Eigen::MatrixXd::Zero(20, 20);
    Eigen::MatrixXd img_next = Eigen::MatrixXd::Zero(20, 20);
    
    vision::TrackedFeature feat(Eigen::Vector2d(0.0, 0.0), Eigen::Vector2d(0.0, 0.0), false); 
    std::vector<vision::TrackedFeature> features = {feat};

    // 2. Act
    vision::calcOpticalFlowLK(img_prev, img_next, features, 3);

    // 3. Assert
    EXPECT_TRUE(features[0].is_lost) << "Точка має бути позначена як втрачена через вихід за межі";
}

// 3. ТЕСТ НА ВИРОДЖЕНУ МАТРИЦЮ (Flat Region / Zero Determinant)
TEST(OpticalFlowTest, CalcOpticalFlowLK_FlatTexturelessRegion_MarksAsLost) {
    // 1. Arrange
    Eigen::MatrixXd img_prev = Eigen::MatrixXd::Constant(20, 20, 0.5);
    Eigen::MatrixXd img_next = Eigen::MatrixXd::Constant(20, 20, 0.5);

    vision::TrackedFeature feat(Eigen::Vector2d(10.0, 10.0), Eigen::Vector2d(10.0, 10.0), false);
    std::vector<vision::TrackedFeature> features = {feat};

    // 2. Act
    vision::calcOpticalFlowLK(img_prev, img_next, features, 3);

    // 3. Assert
    EXPECT_TRUE(features[0].is_lost) << "Точка має бути втрачена через вироджений Гессіан (det < 1e-6)";
}
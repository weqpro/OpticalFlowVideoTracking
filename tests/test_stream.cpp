#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>

#include "optical_flow.h" 

Eigen::MatrixXd createTestImage(int rows, int cols, double square_x, double square_y, int square_size) {
    Eigen::MatrixXd img = Eigen::MatrixXd::Zero(rows, cols);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            // Перевіряємо, чи піксель потрапляє в межі квадрата з дробовими координатами
            if (x >= square_x && x < square_x + square_size &&
                y >= square_y && y < square_y + square_size) {
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
    int win_size = 7; 
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

    double tol = 0.1; 
    EXPECT_NEAR(my_features[0].current_pos.x(), cv_next_pts[0].x, tol);
    EXPECT_NEAR(my_features[0].current_pos.y(), cv_next_pts[0].y, tol);
}

// 2. ТЕСТ НА СУБПІКСЕЛЬНУ ТОЧНІСТЬ
TEST(OpticalFlowTest, CalcOpticalFlowLK_SubpixelShift_ReturnsAccuratePosition) {
    // 1. Arrange
    int rows = 50; int cols = 50;
    Eigen::MatrixXd img_prev = createTestImage(rows, cols, 20, 20, 10);
    Eigen::MatrixXd img_next = createTestImage(rows, cols, 20.3, 20, 10); 

    vision::TrackedFeature feat(Eigen::Vector2d(25.0, 25.0), Eigen::Vector2d(25.0, 25.0), false);
    std::vector<vision::TrackedFeature> features = {feat};

    // 2. Act
    vision::calcOpticalFlowLK(img_prev, img_next, features, 7);

    // 3. Assert
    ASSERT_FALSE(features[0].is_lost) << "Точка не повинна бути втрачена при малому зсуві";
    EXPECT_NEAR(features[0].current_pos.x(), 25.3, 0.05) << "Має бути знайдено субпіксельний зсув по X";
    EXPECT_NEAR(features[0].current_pos.y(), 25.0, 0.05) << "Зсув по Y має бути близьким до нуля";
}

// 3. ТЕСТ НА ВИХІД ЗА МЕЖІ (Boundary Check)
TEST(OpticalFlowTest, CalcOpticalFlowLK_FeatureNearEdge_MarksAsLost) {
    // 1. Arrange
    Eigen::MatrixXd img_prev = Eigen::MatrixXd::Zero(50, 50);
    Eigen::MatrixXd img_next = Eigen::MatrixXd::Zero(50, 50);

    vision::TrackedFeature feat(Eigen::Vector2d(1.0, 25.0), Eigen::Vector2d(1.0, 25.0), false); 
    std::vector<vision::TrackedFeature> features = {feat};

    // 2. Act
    vision::calcOpticalFlowLK(img_prev, img_next, features, 5);

    // 3. Assert
    EXPECT_TRUE(features[0].is_lost) << "Точка має бути позначена як втрачена через близькість до краю";
}

// 4. ТЕСТ НА ВИРОДЖЕНУ МАТРИЦЮ
TEST(OpticalFlowTest, CalcOpticalFlowLK_FlatTexturelessRegion_MarksAsLost) {
    // 1. Arrange
    Eigen::MatrixXd img_prev = Eigen::MatrixXd::Constant(50, 50, 0.5);
    Eigen::MatrixXd img_next = Eigen::MatrixXd::Constant(50, 50, 0.6);

    vision::TrackedFeature feat(Eigen::Vector2d(25.0, 25.0), Eigen::Vector2d(25.0, 25.0), false);
    std::vector<vision::TrackedFeature> features = {feat};

    // 2. Act
    vision::calcOpticalFlowLK(img_prev, img_next, features, 7);

    // 3. Assert
    EXPECT_TRUE(features[0].is_lost) << "Точка має бути втрачена на однорідному фоні (нульовий градієнт)";
}

// 5. ІНТЕГРАЦІЙНИЙ ТЕСТ: ТРЕКІНГ ПОТОКУ
TEST(OpticalFlowStreamTest, SubpixelMultiFrameTracking_MovingObject_TracksAccurately) {
    // 1. Arrange
    const int num_frames = 5;
    const Eigen::Vector2d velocity(1.2, 0.8);
    Eigen::Vector2d ground_truth(20.0, 20.0);
    
    vision::TrackedFeature feat(ground_truth, ground_truth, false);
    std::vector<vision::TrackedFeature> features = {feat};
    Eigen::MatrixXd img_prev = createTestImage(100, 100, ground_truth.x(), ground_truth.y(), 10);

    // 2. Act & 3. Assert (циклічна перевірка)
    for (int f = 0; f < num_frames; ++f) {
        ground_truth += velocity;
        Eigen::MatrixXd img_next = createTestImage(100, 100, ground_truth.x(), ground_truth.y(), 10);

        vision::calcOpticalFlowLK(img_prev, img_next, features, 9);

        ASSERT_FALSE(features[0].is_lost) << "Втрачено на кадрі " << f;
        EXPECT_NEAR(features[0].current_pos.x(), ground_truth.x(), 0.1) << "Помилка по X на кадрі " << f;
        EXPECT_NEAR(features[0].current_pos.y(), ground_truth.y(), 0.1) << "Помилка по Y на кадрі " << f;

        features[0].previous_pos = features[0].current_pos;
        img_prev = img_next;
    }
}
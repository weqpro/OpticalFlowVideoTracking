#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <vector>
#include <algorithm>

#include "optical_flow.h" 

Eigen::MatrixXd createTestImage(int rows, int cols, double square_x, double square_y, int square_size) {
    Eigen::MatrixXd img = Eigen::MatrixXd::Zero(rows, cols);
    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            double x_start = std::max(static_cast<double>(x), square_x);
            double x_end = std::min(static_cast<double>(x + 1), square_x + square_size);
            double y_start = std::max(static_cast<double>(y), square_y);
            double y_end = std::min(static_cast<double>(y + 1), square_y + square_size);
            
            if (x_start < x_end && y_start < y_end) {
                double area = (x_end - x_start) * (y_end - y_start);
                img(y, x) = area;
            }
        }
    }
    return img;
}

// 1. ТЕСТ НА ЗБІГ З OPENCV
TEST(OpticalFlowTest, CalcOpticalFlowLK_SimpleTranslation_MatchesOpenCV) {
    int rows = 50; 
    int cols = 50; 
    int win_size = 7; 
    Eigen::MatrixXd img_prev = createTestImage(rows, cols, 20.0, 20.0, 10);
    Eigen::MatrixXd img_next = createTestImage(rows, cols, 21.0, 21.0, 10);

    vision::TrackedFeature my_feat(Eigen::Vector2d(20.0, 20.0), Eigen::Vector2d(20.0, 20.0), false);
    std::vector<vision::TrackedFeature> my_features = {my_feat};

    cv::Mat cv_img_prev_64, cv_img_next_64;
    cv::eigen2cv(img_prev, cv_img_prev_64);
    cv::eigen2cv(img_next, cv_img_next_64);
    
    cv::Mat cv_img_prev, cv_img_next;
    cv_img_prev_64.convertTo(cv_img_prev, CV_8U, 255.0); 
    cv_img_next_64.convertTo(cv_img_next, CV_8U, 255.0);

    std::vector<cv::Point2f> cv_prev_pts = { cv::Point2f(20.0f, 20.0f) };
    std::vector<cv::Point2f> cv_next_pts;
    std::vector<uchar> status;
    std::vector<float> err;

    // Act
    vision::calcOpticalFlowLK(img_prev, img_next, my_features, win_size);
    cv::calcOpticalFlowPyrLK(cv_img_prev, cv_img_next, cv_prev_pts, cv_next_pts, 
                             status, err, cv::Size(win_size, win_size), 0);

    // Assert
    ASSERT_FALSE(my_features[0].is_lost) << "Точка не повинна бути втрачена";
    ASSERT_EQ(status[0], 1) << "OpenCV також повинен знайти точку";

    double tol = 0.2; 
    EXPECT_NEAR(my_features[0].current_pos.x(), cv_next_pts[0].x, tol);
    EXPECT_NEAR(my_features[0].current_pos.y(), cv_next_pts[0].y, tol);
}

// 2. ТЕСТ НА СУБПІКСЕЛЬНУ ТОЧНІСТЬ
TEST(OpticalFlowTest, CalcOpticalFlowLK_SubpixelShift_ReturnsAccuratePosition) {
    int rows = 50; int cols = 50;
    Eigen::MatrixXd img_prev = createTestImage(rows, cols, 20.0, 20.0, 10);
    Eigen::MatrixXd img_next = createTestImage(rows, cols, 20.3, 20.0, 10); 

    vision::TrackedFeature feat(Eigen::Vector2d(20.0, 20.0), Eigen::Vector2d(20.0, 20.0), false);
    std::vector<vision::TrackedFeature> features = {feat};

    vision::calcOpticalFlowLK(img_prev, img_next, features, 7);

    ASSERT_FALSE(features[0].is_lost) << "Точка не повинна бути втрачена при малому зсуві";
    EXPECT_NEAR(features[0].current_pos.x(), 20.3, 0.1) << "Має бути знайдено субпіксельний зсув по X";
    EXPECT_NEAR(features[0].current_pos.y(), 20.0, 0.1) << "Зсув по Y має бути близьким до нуля";
}

// 3. ТЕСТ НА ВИХІД ЗА МЕЖІ (Boundary Check)
TEST(OpticalFlowTest, CalcOpticalFlowLK_FeatureNearEdge_MarksAsLost) {
    Eigen::MatrixXd img_prev = Eigen::MatrixXd::Zero(50, 50);
    Eigen::MatrixXd img_next = Eigen::MatrixXd::Zero(50, 50);

    vision::TrackedFeature feat(Eigen::Vector2d(1.0, 25.0), Eigen::Vector2d(1.0, 25.0), false); 
    std::vector<vision::TrackedFeature> features = {feat};

    vision::calcOpticalFlowLK(img_prev, img_next, features, 5);

    EXPECT_TRUE(features[0].is_lost) << "Точка має бути позначена як втрачена через близькість до краю";
}

// 4. ТЕСТ НА ВИРОДЖЕНУ МАТРИЦЮ
TEST(OpticalFlowTest, CalcOpticalFlowLK_FlatTexturelessRegion_MarksAsLost) {
    Eigen::MatrixXd img_prev = Eigen::MatrixXd::Constant(50, 50, 0.5);
    Eigen::MatrixXd img_next = Eigen::MatrixXd::Constant(50, 50, 0.6);

    vision::TrackedFeature feat(Eigen::Vector2d(25.0, 25.0), Eigen::Vector2d(25.0, 25.0), false);
    std::vector<vision::TrackedFeature> features = {feat};

    vision::calcOpticalFlowLK(img_prev, img_next, features, 7);

    EXPECT_TRUE(features[0].is_lost) << "Точка має бути втрачена на однорідному фоні";
}

// 5. ІНТЕГРАЦІЙНИЙ ТЕСТ: ТРЕКІНГ ПОТОКУ
TEST(OpticalFlowStreamTest, SubpixelMultiFrameTracking_MovingObject_TracksAccurately) {
    const int num_frames = 5;
    const Eigen::Vector2d velocity(0.5, 0.3);
    Eigen::Vector2d ground_truth(20.0, 20.0);
    
    vision::TrackedFeature feat(ground_truth, ground_truth, false);
    std::vector<vision::TrackedFeature> features = {feat};
    Eigen::MatrixXd img_prev = createTestImage(100, 100, ground_truth.x(), ground_truth.y(), 10);

    for (int f = 0; f < num_frames; ++f) {
        ground_truth += velocity;
        Eigen::MatrixXd img_next = createTestImage(100, 100, ground_truth.x(), ground_truth.y(), 10);

        vision::calcOpticalFlowLK(img_prev, img_next, features, 9);

        ASSERT_FALSE(features[0].is_lost) << "Втрачено на кадрі " << f;
        EXPECT_NEAR(features[0].current_pos.x(), ground_truth.x(), 0.2) << "Помилка по X на кадрі " << f;
        EXPECT_NEAR(features[0].current_pos.y(), ground_truth.y(), 0.2) << "Помилка по Y на кадрі " << f;

        features[0].previous_pos = features[0].current_pos;
        img_prev = img_next;
    }
}

// 6. ТЕСТ НА ВЕЛИКЕ ЗМІЩЕННЯ (Multi-level)
TEST(OpticalFlowTest, CalcOpticalFlowLK_LargeDisplacement_TracksWithPyramid) {
    int rows = 100; 
    int cols = 100; 
    // Зміщення на 8 пікселів - забагато для стандартного вікна 7x7 без піраміди
    Eigen::MatrixXd img_prev = createTestImage(rows, cols, 40.0, 40.0, 15);
    Eigen::MatrixXd img_next = createTestImage(rows, cols, 48.0, 48.0, 15);

    vision::TrackedFeature feat(Eigen::Vector2d(40.0, 40.0), Eigen::Vector2d(40.0, 40.0), false);
    std::vector<vision::TrackedFeature> features = {feat};

    // Спочатку спробуємо без пірамід (має бути неточним для великого зсуву)
    vision::calcOpticalFlowLK(img_prev, img_next, features, 7, 1);
    double dist_single = (features[0].current_pos - Eigen::Vector2d(48.0, 48.0)).norm();

    // Тепер з 3 рівнями піраміди
    features[0].current_pos = features[0].previous_pos;
    features[0].is_lost = false;
    vision::calcOpticalFlowLK(img_prev, img_next, features, 7, 3);
    double dist_multi = (features[0].current_pos - Eigen::Vector2d(48.0, 48.0)).norm();

    EXPECT_LT(dist_multi, 0.5) << "Багатомасштабний підхід має знайти точку з високою точністю";
    EXPECT_GT(dist_single, dist_multi) << "Багатомасштабний підхід має бути кращим за одномасштабний для великих зсувів";
}

// 7. ТЕСТ НА РОБАСТНІСТЬ (Robustness to Outliers)
TEST(OpticalFlowTest, CalcOpticalFlowLK_WithOutliers_TracksCorrectly) {
    int rows = 50; 
    int cols = 50; 
    Eigen::MatrixXd img_prev = createTestImage(rows, cols, 20.0, 20.0, 10);
    Eigen::MatrixXd img_next = createTestImage(rows, cols, 21.0, 20.0, 10);

    // Додаємо "шум" (викид) поруч з кутом на другому кадрі
    img_next(22, 22) = 5.0; 

    vision::TrackedFeature feat(Eigen::Vector2d(20.0, 20.0), Eigen::Vector2d(20.0, 20.0), false);
    std::vector<vision::TrackedFeature> features = {feat};

    // З IRLS алгоритм має ігнорувати цей піксель і знайти правильний зсув (1.0, 0.0)
    vision::calcOpticalFlowLK(img_prev, img_next, features, 7, 1);

    ASSERT_FALSE(features[0].is_lost) << "Точка не повинна бути втрачена через один викид";
    EXPECT_NEAR(features[0].current_pos.x(), 21.0, 0.2) << "Має бути знайдено правильний зсув по X попри викид";
    EXPECT_NEAR(features[0].current_pos.y(), 20.0, 0.2) << "Зсув по Y має залишатися близьким до нуля";
}

// 8. ТЕСТ НА СТІЙКІСТЬ ДО ОСВІТЛЕННЯ (Lighting Invariance)
TEST(OpticalFlowTest, CalcOpticalFlowLK_VariableLighting_TracksCorrectly) {
    int rows = 100; 
    int cols = 100; 
    Eigen::MatrixXd img_prev = createTestImage(rows, cols, 40.0, 40.0, 15);
    
    // Другий кадр має зсув на 1 піксель та значно яскравіший (+0.5 фонового світла)
    Eigen::MatrixXd img_next = createTestImage(rows, cols, 41.0, 40.0, 15);
    img_next = img_next.array() + 0.5;

    vision::TrackedFeature feat(Eigen::Vector2d(40.0, 40.0), Eigen::Vector2d(40.0, 40.0), false);
    std::vector<vision::TrackedFeature> features = {feat};

    // Завдяки нормалізації алгоритм має знайти зсув (1.0, 0.0) попри зміну яскравості
    vision::calcOpticalFlowLK(img_prev, img_next, features, 7, 1);

    ASSERT_FALSE(features[0].is_lost) << "Точка не повинна бути втрачена через зміну освітлення";
    EXPECT_NEAR(features[0].current_pos.x(), 41.0, 0.2) << "Має бути знайдено правильний зсув по X попри яскравість";
    EXPECT_NEAR(features[0].current_pos.y(), 40.0, 0.2) << "Зсув по Y має бути близьким до нуля";
}

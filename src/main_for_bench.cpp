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

#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <filesystem>
#include <Eigen/Dense>
#include "optical_flow.h"
#include "feature_detector.h"
#include "video/stream.h"

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_sequence_pattern_or_video>" << std::endl;
        std::cerr << "Example for images: " << argv[0] << " \"img/frame_%04d.png\"" << std::endl;
        return 1;
    }

    try {
        video::Stream video_stream(argv[1]);
        
        auto frame_prev_opt = video_stream.getFrame();
        if (!frame_prev_opt) {
            std::cerr << "Error: Could not load the first image/frame." << std::endl;
            return 1;
        }

        Eigen::MatrixXd frame_prev = *frame_prev_opt;

        std::vector<Eigen::Vector2d> corners = vision::findGoodFeaturesToTrack(frame_prev, 150, 0.01, 10.0);
        std::vector<vision::TrackedFeature> features;
        for (const auto& p : corners) {
            features.emplace_back(p, p);
        }

        std::cout << "--- Dataset Evaluation Started ---" << std::endl;
        std::cout << "Initial features detected: " << features.size() << std::endl;
        std::cout << "Image resolution: " << frame_prev.cols() << "x" << frame_prev.rows() << std::endl;

        int frame_count = 0;
        double total_latency = 0;

        while (true) {
            auto frame_next_opt = video_stream.getFrame();
            if (!frame_next_opt) break;

            Eigen::MatrixXd frame_next = *frame_next_opt;

            auto start = std::chrono::high_resolution_clock::now();

            vision::calcOpticalFlowLK(frame_prev, frame_next, features, 7, 3);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;
            
            total_latency += elapsed.count();
            frame_count++;

            int tracked = 0;
            double avg_flow_mag = 0;
            for (auto& feat : features) {
                if (!feat.is_lost) {
                    avg_flow_mag += (feat.current_pos - feat.previous_pos).norm();
                    feat.previous_pos = feat.current_pos;
                    tracked++;
                }
            }
            if (tracked > 0) avg_flow_mag /= tracked;

            if (frame_count % 5 == 0) {
                std::cout << "[Step " << std::setw(3) << frame_count << "] "
                          << "Tracked: " << std::setw(3) << tracked << " | "
                          << "Latency: " << std::fixed << std::setprecision(2) << elapsed.count() * 1000.0 << "ms | "
                          << "Avg Motion: " << avg_flow_mag << " px" << std::endl;
            }

            if (tracked < 30) {
                auto new_corners = vision::findGoodFeaturesToTrack(frame_next, 150, 0.01, 10.0);
                for (const auto& p : new_corners) {
                    features.emplace_back(p, p);
                }
            }

            frame_prev = std::move(frame_next);
        }

        if (frame_count > 0) {
            double avg_lat = (total_latency / frame_count) * 1000.0;
            std::cout << "\n===============================" << std::endl;
            std::cout << "FINAL PERFORMANCE REPORT" << std::endl;
            std::cout << "-------------------------------" << std::endl;
            std::cout << "Total images processed: " << frame_count << std::endl;
            std::cout << "Average Latency:        " << avg_lat << " ms" << std::endl;
            std::cout << "Theoretical Max FPS:    " << 1000.0 / avg_lat << std::endl;
            std::cout << "Algorithm Stability:    " << "Robust (IRLS enabled)" << std::endl;
            std::cout << "===============================" << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

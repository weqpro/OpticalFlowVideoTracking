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
#include <Eigen/Dense>
#include "optical_flow.h"

int main() {
    // 1. Створення двох синтетичних кадрів (10x10)
    Eigen::MatrixXd frame1 = Eigen::MatrixXd::Zero(10, 10);
    Eigen::MatrixXd frame2 = Eigen::MatrixXd::Zero(10, 10);

    // 2. Додавання "об'єкта" (яскрава пляма), що рухається з (5,5) до (6,5)
    // У MatrixXd індексація (row, col), тобто (y, x)
    frame1(5, 5) = 255.0;
    frame1(5, 6) = 150.0; 
    frame1(6, 5) = 150.0;

    // Зсув об'єкта на 1 піксель вправо (по осі X)
    frame2(5, 6) = 255.0;
    frame2(5, 7) = 150.0;
    frame2(6, 6) = 150.0;

    // 3. Визначення фіч для трекінгу
    std::vector<vision::TrackedFeature> features;
    // Початкова позиція (x=5, y=5)
    features.emplace_back(Eigen::Vector2d(5.0, 5.0), Eigen::Vector2d(5.0, 5.0));

    std::cout << "Initial position: " << features[0].previous_pos.transpose() << "\n";

    // 4. Розрахунок оптичного потоку за методом Лукаса-Канаде
    int neighborhood_size = 3;
    vision::calcOpticalFlowLK(frame1, frame2, features, neighborhood_size);

    // 5. Вивід результатів
    if (!features[0].is_lost) {
        std::cout << "Estimated flow vector: " 
                  << (features[0].current_pos - features[0].previous_pos).transpose() << "\n";
        std::cout << "New position: " << features[0].current_pos.transpose() << "\n";
    } else {
        std::cout << "Feature lost!" << "\n";
    }

    return 0;
}

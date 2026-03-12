#include <iostream>
#include <stdexcept>
#include "video/stream.h"

int main() {
    // --- Case 1: bad path must throw, not crash ---
    try {
        video::Stream bad("/tmp/does_not_exist.mp4");
        std::cerr << "FAIL: expected exception for bad path\n";
        return 1;
    } catch (const std::runtime_error& e) {
        std::cout << "OK bad-path throw: " << e.what() << "\n";
    }

    // --- Case 2: good path — iterate all frames ---
    try {
        video::Stream s("/tmp/test.mp4");
        int count = 0;
        int rows = 0, cols = 0;
        while (auto frame = s.getFrame()) {
            if (count == 0) { rows = frame->rows(); cols = frame->cols(); }
            ++count;
        }
        std::cout << "OK decoded " << count << " frames, each "
                  << rows << "x" << cols << " (double, [0,1])\n";
        if (rows != 240 || cols != 320) {
            std::cerr << "FAIL: expected 240x320\n";
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

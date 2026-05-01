# OpticalFlowVideoTracking

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for the full license text.

Copyright 2026 Konovalenko Stanislav and Hombosh Oleh

## Video

- Stanislav Konovalenko: [YouTube](https://youtu.be/mMappyirUGM?si=1NfK5A0wVT_a-lS0)
- Hombosh Oleh: [YouTube](https://youtu.be/qdFO_dptG9Q)

## Prerequisites

- C++ Compiler: Visual Studio 2019+ (MSVC) or MinGW-w64 (with C++20 support)
- CMake: 3.16+
- Eigen3: Library for linear algebra
- FFmpeg: Must be installed and added to the system variable FFMPEG_PATH (e.g., C:\ffmpeg)


###  Building

  Dependencies: CMake 3.16+, Eigen3, FFmpeg (libavformat, libavcodec, libavutil, libswscale), OpenCV

  # Release build
  ./compile.sh

  # Debug build
  ./compile.sh -d

  # Build and run tests
  ./compile.sh -t

  Binaries are written to build/bin/.

###  Running

  Lucas-Kanade optical flow demo (synthetic frames):

  ./build/bin/OpticalFlowVideoTracking

  Flow visualizer (generates flow_input.mp4, processes it with feature tracking, writes result to flow_output.mp4):

  ./build/bin/flow_visualizer

  Run tests only (after building):

  ctest --test-dir build --output-on-failure


## Installation

```
git clone https://github.com/weqpro/OpticalFlowVideoTracking
cd OpticalFlowVideoTracking
```


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

## Compilation

For Windows (using PowerShell or Command Prompt):

```sh
mkdir build
cd build
cmake .. -DENABLE_CLANG_TIDY=OFF
cmake --build . --config Release
```

Note: FFmpeg files (.dll) will be automatically copied to the executable directory after a successful build.

## Installation

```
git clone https://github.com/weqpro/OpticalFlowVideoTracking
cd OpticalFlowVideoTracking
```

## Usage

After successful compilation, the executable file will be located in the build/bin/Release folder (or just build/bin). To run the program, use the following command:

```sh
.\build\bin\Release\OpticalFlowVideoTracking.exe
```

Example output:

1 Initial position: 5 5
2 Estimated flow vector: 1 0

# OpticalFlowVideoTracking

## License

This project is licensed under the Apache License, Version 2.0. See the [LICENSE](LICENSE) file for the full license text.

Copyright 2026 Konovalenko Stanislav and Hombosh Oleh

## Video

- Stanislav Konovalenko: [YouTube](https://youtu.be/mMappyirUGM?si=1NfK5A0wVT_a-lS0)
- Hombosh Oleh: [YouTube](https://youtu.be/qdFO_dptG9Q)

## Prerequisites

- C++ Compiler: Visual Studio 2019+ (MSVC) або MinGW-w64 (з підтримкою C++20)
- CMake: 3.16+
- Eigen3: Бібліотека для лінійної алгебри
- FFmpeg: Потрібно встановити та додати шлях до системної змінної FFMPEG_PATH (наприклад, C:\ffmpeg)

## Compilation

Для Windows (використовуючи PowerShell або Command Prompt):

```sh
mkdir build
cd build
cmake .. -DENABLE_CLANG_TIDY=OFF
cmake --build . --config Release
```

Примітка: Файли FFmpeg (.dll) будуть автоматично скопійовані в директорію з виконуваним файлом після успішної збірки.

## Installation

```
git clone https://github.com/weqpro/OpticalFlowVideoTracking
cd OpticalFlowVideoTracking
```

## Usage

Після успішної компіляції виконуваний файл буде знаходитись у папці build/bin/Release (або просто
build/bin). Для запуску програми скористайтесь командою:

```sh
.\build\bin\Release\OpticalFlowVideoTracking.exe
```

Приклад виводу:

1 Initial position: 5 5
2 Estimated flow vector: 1 0


mkdir -p build
cmake -B build -DENABLE_CLANG_TIDY=ON && cmake --build build

mkdir -p build
cmake -B build -DENABLE_CLANG_TIDY=ON && cmake --build build || exit 1

if [[ "$*" == *"--test"* ]]; then
    ctest --test-dir build --output-on-failure
fi

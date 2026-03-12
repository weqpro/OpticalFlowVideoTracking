mkdir -p build && cd build || exit 1
cmake .. -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build .
cd ..

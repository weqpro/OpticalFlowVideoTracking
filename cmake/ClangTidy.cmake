# cmake/ClangTidy.cmake
# ─────────────────────────────────────────────────────────────────────────────
# Opt-in clang-tidy integration.
#
# CMake property docs:
#   https://cmake.org/cmake/help/latest/prop_tgt/LANG_CLANG_TIDY.html
#
# Enable:
#   cmake -B build -DENABLE_CLANG_TIDY=ON
#   cmake -B build -DENABLE_CLANG_TIDY=ON -DCLANG_TIDY_WARNINGS_AS_ERRORS=ON
# ─────────────────────────────────────────────────────────────────────────────

option(ENABLE_CLANG_TIDY             "Run clang-tidy alongside compilation"  OFF)
option(CLANG_TIDY_WARNINGS_AS_ERRORS "Treat clang-tidy findings as errors"   OFF)

if(NOT ENABLE_CLANG_TIDY)
  return()
endif()

find_program(CLANG_TIDY_EXE NAMES "clang-tidy")
if(NOT CLANG_TIDY_EXE)
  message(WARNING "ENABLE_CLANG_TIDY=ON but clang-tidy not found — skipping.")
  return()
endif()

message(STATUS "clang-tidy: ${CLANG_TIDY_EXE}")

# Build the command list.
# --header-filter: only flag issues in project headers (include/).
#   Third-party headers (FFmpeg, Eigen) are excluded — they contain C-isms
#   that generate hundreds of cppcoreguidelines false positives.
#   Pattern used by google-cloud-cpp and LLVM.
set(_clang_tidy_cmd
  "${CLANG_TIDY_EXE}"
  "--header-filter=^${CMAKE_SOURCE_DIR}/include/.*"
  "--use-color"
)

if(CLANG_TIDY_WARNINGS_AS_ERRORS)
  list(APPEND _clang_tidy_cmd "--warnings-as-errors=*")
endif()

# Apply clang-tidy to a single CMake target.
# Uses CXX_CLANG_TIDY (per-target) rather than CMAKE_CXX_CLANG_TIDY (global)
# so future targets (e.g. tests) can use different rules or opt out.
function(target_enable_clang_tidy target)
  set_target_properties("${target}" PROPERTIES
    CXX_CLANG_TIDY "${_clang_tidy_cmd}"
  )
endfunction()

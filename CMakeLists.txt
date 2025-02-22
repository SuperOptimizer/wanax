cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

project(wanax C CXX)

# Set C standard
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Adjust paths for new structure
set(PROJECT_ROOT .)
set(BUILD_ROOT ${PROJECT_ROOT}/build)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${BUILD_ROOT}/cmake-build-${CMAKE_BUILD_TYPE}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${BUILD_ROOT}/cmake-build-${CMAKE_BUILD_TYPE}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${BUILD_ROOT}/cmake-build-${CMAKE_BUILD_TYPE}/lib)

# Compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic -Weverything")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -g -O0")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3")

# Sanitizer options for Debug builds
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fsanitize=address,undefined,leak -fno-omit-frame-pointer")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS_DEBUG} -fsanitize=address,undefined,leak")

add_executable(wanax main.c wanax.c wanax.h)

enable_testing()
add_executable(wanax_test test.c wanax.c wanax.h)
add_test(NAME wanax_test COMMAND wanax_test)

install(TARGETS wanax
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
)
install(FILES ${LIB_HEADERS} DESTINATION include)
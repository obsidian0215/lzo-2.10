cmake_minimum_required(VERSION 3.10)
project(lzo_cpu_cli C)

# Sources derived from the successful gcc command
set(LZO_CPU_SOURCES
    lzo_frag.c
    lzo1x_1.c
    lzo1x_1k.c
    lzo1x_1l.c
    lzo1x_1o.c
    lzo1x_d1.c
    src/lzo_init.c
    src/lzo_ptr.c
    src/lzo_str.c
    src/lzo_util.c
    src/lzo_crc.c
)

add_executable(lzo_cpu ${LZO_CPU_SOURCES})

# Include paths relative to this subdirectory
# - .               (for local headers like lzo_levels.h, config1x.h)
# - ./src           (for internal LZO headers in this local copy)
# - ./include       (public include)
# - ./include/lzo   (for lzoconf.h etc when headers are included as <lzo/...>)
target_include_directories(lzo_cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/lzo
)

# Require C11 (equivalent to -std=c11)
set_property(TARGET lzo_cpu PROPERTY C_STANDARD 11)
set_property(TARGET lzo_cpu PROPERTY C_STANDARD_REQUIRED ON)

# Optimize in Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Threads (maps to -pthread on MinGW)
find_package(Threads REQUIRED)
if(Threads_FOUND)
    target_link_libraries(lzo_cpu PRIVATE Threads::Threads)
endif()

# Helpful warnings on GCC/Clang
if(CMAKE_C_COMPILER_ID MATCHES "^(GNU|Clang)$")
    target_compile_options(lzo_cpu PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Builds the C example against the emberwave shared library that
# `cargo build -p emberwave-ffi` makes.
#
#     cargo build --release -p emberwave-ffi
#     cmake -S crates/emberwave-ffi/examples/c -B build/ffi-example -DEMBERWAVE_LIB_DIR=target/release
#     cmake --build build/ffi-example --config Release
#
# EMBERWAVE_LIB_DIR is the cargo output directory (target/debug or
# target/release), relative to the repository root or absolute.
cmake_minimum_required(VERSION 3.16)
project(emberwave_example C)

set(EMBERWAVE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
set(EMBERWAVE_LIB_DIR "${EMBERWAVE_ROOT}/target/release" CACHE PATH "Where cargo put the emberwave-ffi library")
get_filename_component(EMBERWAVE_LIB_DIR "${EMBERWAVE_LIB_DIR}" ABSOLUTE BASE_DIR "${EMBERWAVE_ROOT}")

add_library(emberwave SHARED IMPORTED)
set_target_properties(emberwave PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
if(WIN32)
    set_target_properties(emberwave PROPERTIES
        IMPORTED_LOCATION "${EMBERWAVE_LIB_DIR}/emberwave_ffi.dll"
        IMPORTED_IMPLIB "${EMBERWAVE_LIB_DIR}/emberwave_ffi.dll.lib")
elseif(APPLE)
    set_target_properties(emberwave PROPERTIES
        IMPORTED_LOCATION "${EMBERWAVE_LIB_DIR}/libemberwave_ffi.dylib")
else()
    set_target_properties(emberwave PROPERTIES
        IMPORTED_LOCATION "${EMBERWAVE_LIB_DIR}/libemberwave_ffi.so")
endif()

add_executable(emberwave_decode decode.c)
target_link_libraries(emberwave_decode PRIVATE emberwave)
if(MSVC)
    target_compile_options(emberwave_decode PRIVATE /W4 /WX)
    # fopen is standard C; MSVC would rather have fopen_s.
    target_compile_definitions(emberwave_decode PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(emberwave_decode PRIVATE -std=c99 -Wall -Wextra -Werror -pedantic)
endif()

# The library is found next to the executable on Windows, and through the
# run path (the library's directory) elsewhere.
if(WIN32)
    add_custom_command(TARGET emberwave_decode POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${EMBERWAVE_LIB_DIR}/emberwave_ffi.dll" "$<TARGET_FILE_DIR:emberwave_decode>")
else()
    set_target_properties(emberwave_decode PROPERTIES BUILD_RPATH "${EMBERWAVE_LIB_DIR}")
endif()

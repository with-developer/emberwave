# Builds the C examples against the emberwave shared library that
# `cargo build -p emberwave-ffi` makes: emberwave_decode (decode.c) and
# emberwave_encode (encode.c).
#
#     cargo build --release -p emberwave-ffi
#     cmake -S crates/emberwave-ffi/examples/c -B target/c-example -DEMBERWAVE_LIB_DIR=target/release
#     cmake --build target/c-example --config Release
#
# EMBERWAVE_LIB_DIR is the cargo output directory (target/debug or
# target/release), relative to the repository root or absolute.
cmake_minimum_required(VERSION 3.16)
project(emberwave_example C)

set(EMBERWAVE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../..")
# A STRING, not a PATH: CMake would resolve a relative PATH against the
# current directory before this file sees it.
set(EMBERWAVE_LIB_DIR "target/release" CACHE STRING "Where cargo put the emberwave-ffi library")
get_filename_component(EMBERWAVE_LIB_DIR_ABSOLUTE "${EMBERWAVE_LIB_DIR}" ABSOLUTE BASE_DIR "${EMBERWAVE_ROOT}")

add_library(emberwave SHARED IMPORTED)
set_target_properties(emberwave PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../../include")
if(WIN32)
    set_target_properties(emberwave PROPERTIES
        IMPORTED_LOCATION "${EMBERWAVE_LIB_DIR_ABSOLUTE}/emberwave_ffi.dll"
        IMPORTED_IMPLIB "${EMBERWAVE_LIB_DIR_ABSOLUTE}/emberwave_ffi.dll.lib")
elseif(APPLE)
    set_target_properties(emberwave PROPERTIES
        IMPORTED_LOCATION "${EMBERWAVE_LIB_DIR_ABSOLUTE}/libemberwave_ffi.dylib")
else()
    set_target_properties(emberwave PROPERTIES
        IMPORTED_LOCATION "${EMBERWAVE_LIB_DIR_ABSOLUTE}/libemberwave_ffi.so")
endif()

foreach(example emberwave_decode emberwave_encode)
    string(REPLACE "emberwave_" "" source ${example})
    add_executable(${example} ${source}.c)
    target_link_libraries(${example} PRIVATE emberwave)
    # The header is included with -I rather than as a system header, so that
    # the warnings below check it too.
    set_target_properties(${example} PROPERTIES NO_SYSTEM_FROM_IMPORTED ON)
    if(MSVC)
        target_compile_options(${example} PRIVATE /W4 /WX)
        # fopen and strtok are standard C; MSVC would rather have its _s
        # versions.
        target_compile_definitions(${example} PRIVATE _CRT_SECURE_NO_WARNINGS)
    else()
        target_compile_options(${example} PRIVATE -std=c99 -Wall -Wextra -Werror -pedantic)
        # log10, for the PSNR of encode.c.
        target_link_libraries(${example} PRIVATE m)
    endif()
endforeach()

# The library is found next to the executable on Windows, and through the
# run path (the library's directory) elsewhere.
if(WIN32)
    # Copied on every build, not only when the program is linked again: a
    # rebuilt library with the same exports does not relink the program.
    add_custom_target(emberwave_dll ALL
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${EMBERWAVE_LIB_DIR_ABSOLUTE}/emberwave_ffi.dll" "$<TARGET_FILE_DIR:emberwave_decode>")
    add_dependencies(emberwave_dll emberwave_decode emberwave_encode)
else()
    set_target_properties(emberwave_decode emberwave_encode PROPERTIES
        BUILD_RPATH "${EMBERWAVE_LIB_DIR_ABSOLUTE}")
endif()

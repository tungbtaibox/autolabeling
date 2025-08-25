cmake_minimum_required(VERSION 3.16)
project(autolabel LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Qt ----
find_package(Qt6 REQUIRED COMPONENTS Core Widgets Gui)

# ---- OpenCV (manual) ----
set(OpenCV_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/include")
set(OpenCV_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/x64/vc16/lib")

include_directories(${OpenCV_INCLUDE_DIR})
link_directories(${OpenCV_LIB_DIR})

# Các lib chính của OpenCV (nếu thiếu bạn bổ sung thêm .lib trong thư mục lib)
set(OpenCV_LIBS
    opencv_world4120.lib
    opencv_world4120d.lib
)

# ---- ONNX Runtime ----
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# --- Configure your project files ---
include_directories(include)  # Include your header files directory
# Recursively collect all source files under 'src' directory
file(GLOB_RECURSE CURR_SOURCES src/*.cpp)
# ---- Source ----
add_executable(autolabel
    main.cpp
    ${CURR_SOURCES}
)

# ---- Link libraries ----
target_link_libraries(autolabel
    PRIVATE
        Qt6::Core
        Qt6::Widgets
        Qt6::Gui
        ${OpenCV_LIBS}
        onnxruntime.lib
        onnxruntime_providers_shared.lib
)

# ---- Copy DLLs sau khi build ----
if(WIN32)
    # copy onnxruntime dll
    add_custom_command(TARGET autolabel POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll"
            $<TARGET_FILE_DIR:autolabel>
    )
    add_custom_command(TARGET autolabel POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_DIR}/lib/onnxruntime_providers_shared.dll"
            $<TARGET_FILE_DIR:autolabel>
    )

    # copy OpenCV dlls
    file(GLOB OPENCV_DLLS "${CMAKE_CURRENT_SOURCE_DIR}/opencv/build/x64/vc16/bin/*.dll")
    foreach(opencv_dll ${OPENCV_DLLS})
        add_custom_command(TARGET autolabel POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${opencv_dll}
                $<TARGET_FILE_DIR:autolabel>
        )
    endforeach()
endif()

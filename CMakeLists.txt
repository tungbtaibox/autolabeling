cmake_minimum_required(VERSION 3.16)
project(autolabeling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Qt ----
find_package(Qt6 REQUIRED COMPONENTS Core )
include_directories(include)
file(GLOB_RECURSE CURR_SOURCES src/*.cpp)
# ---- OpenCV ----
set(OpenCV_DIR "${CMAKE_CURRENT_SOURCE_DIR}/opencv")
set(OpenCV_INCLUDE_DIR "${OpenCV_DIR}/include")
set(OpenCV_LIB_DIR "${OpenCV_DIR}/lib")

include_directories(${OpenCV_INCLUDE_DIR})
link_directories(${OpenCV_LIB_DIR})

# Tự động tìm các file .lib trong thư mục
file(GLOB OpenCV_LIBS_RELEASE "${OpenCV_LIB_DIR}/*.so")
set(OpenCV_LIBS ${OpenCV_LIBS_RELEASE})

# ---- ONNXRUNTIME ----
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_DIR}/include/onnxruntime")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_DIR}/lib")

# Set CMAKE_PREFIX_PATH to help find ONNX Runtime
set(CMAKE_PREFIX_PATH ${ONNXRUNTIME_DIR} ${CMAKE_PREFIX_PATH})

# Try to find ONNX Runtime using CMake
find_package(onnxruntime QUIET)
if(onnxruntime_FOUND)
    message(STATUS "Found ONNX Runtime via CMake")
    set(ONNXRUNTIME_LIBS onnxruntime::onnxruntime)
else()
    message(STATUS "ONNX Runtime not found via CMake, using manual linking")
    include_directories(${ONNXRUNTIME_INCLUDE_DIR})
    link_directories(${ONNXRUNTIME_LIB_DIR})

    # Link ONNX Runtime libraries explicitly
    # Note: The C++ API depends on the C API, so we need to link both
    set(ONNXRUNTIME_LIBS
        ${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so
        ${ONNXRUNTIME_LIB_DIR}/libonnxruntime_providers_shared.so
    )

    # Add compiler flags to ensure proper linking
    add_compile_definitions(ORT_API_VERSION=1)

endif()

# ---- Executable ----
add_executable(autolabel
    main.cpp
    ${CURR_SOURCES}
    # Thêm các file source khác ở đây nếu có
)

# ---- Link libraries ----
target_link_libraries(autolabel
    PRIVATE
        Qt6::Core
        ${OpenCV_LIBS}
)

# Link ONNX Runtime separately to ensure proper order
if(onnxruntime_FOUND)
    target_link_libraries(autolabel PRIVATE onnxruntime::onnxruntime)
else()
    target_link_libraries(autolabel PRIVATE ${ONNXRUNTIME_LIBS})
    # ONNX Runtime may need additional system libraries
    target_link_libraries(autolabel PRIVATE pthread dl)
endif()

# Set RPATH to ensure libraries can be found at runtime
set_target_properties(autolabel PROPERTIES
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR};${OpenCV_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
)


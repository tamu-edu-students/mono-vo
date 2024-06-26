# cmake_minimum_required(VERSION 2.8)
cmake_minimum_required(VERSION 3.20)
project( mono-vo )
find_package( OpenCV REQUIRED )
find_package(PkgConfig REQUIRED)
find_package(GDAL REQUIRED)
set(MATIO_LIBRARIES "/usr/local/lib/libmatio.so")
pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories(${GTK3_INCLUDE_DIRS} )
include_directories(${MATIO_INCLUDE_DIRS})
link_directories(${GTK3_LIBRARY_DIRS} )
link_libraries(jsoncpp ${GTK3_LIBRARIES} ${OpenCV_LIBS} ${GDAL_LIBRARY} ${MATIO_LIBRARIES})


set (use_cuvis true) #we should be using cuvis for functions

if (use_cuvis)

    list(APPEND CMAKE_MODULE_PATH "/cuvis.sdk/cuvis.cpp/")
    list(APPEND CMAKE_MODULE_PATH "/cuvis.sdk/cuvis.c/")
    set(DOXYGEN_BUILD_DOCUMENTATION OFF CACHE BOOL "Create and install the HTML based API documentation (requires Doxygen)" FORCE)

    add_definitions(-Duse_cuvis)
    find_package(Cuvis REQUIRED 3.2.0)
    find_package(CuvisCpp REQUIRED)   
    else (use_cuvis)
    
endif()


# torch stuff
list(APPEND CMAKE_MODULE_PATH "/torch/libtorch/share/cmake")
set(Torch_DIR "/torch/libtorch/share/cmake/Torch/")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -lm -ldl")

file(GLOB viso
    "src/*.h"
    "src/*.cpp"
)

if (use_cuvis)
    add_executable(vo ${viso})
    target_link_libraries(vo PRIVATE cuvis::cpp ${OpenCV_LIBS}  ${TORCH_LIBRARIES})
endif()
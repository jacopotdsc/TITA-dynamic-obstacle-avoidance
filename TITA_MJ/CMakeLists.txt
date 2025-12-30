cmake_minimum_required(VERSION 3.16)
project(TITA_MJ)

option(BUILD_PYTHON_BINDINGS "Decide if compile the bindings")

if(BUILD_PYTHON_BINDINGS)
message(STATUS ">>> MODE: PYTHON, compilining wm.so)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${PROJECT_SOURCE_DIR}/compiled;$ENV{CONDA_PREFIX}/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

find_package(glfw3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

#set(CONDA_PREFIX $ENV{CONDA_PREFIX})

find_package(mujoco CONFIG QUIET)
if(TARGET mujoco::mujoco)
    message(STATUS "MuJoCo found: using mujoco::mujoco")
else()
    set(MUJOCO_PATH "$ENV{HOME}/.pixi/build/work/.../host_placehold*/lib")
    find_library(MUJOCO_LIBRARY NAMES mujoco PATHS ${MUJOCO_PATH})
    if(NOT MUJOCO_LIBRARY)
        message(FATAL_ERROR "MuJoCo library not found in ${MUJOCO_PATH}")
    endif()
    add_library(mujoco::mujoco UNKNOWN IMPORTED)
    set_target_properties(mujoco::mujoco PROPERTIES IMPORTED_LOCATION ${MUJOCO_LIBRARY})
endif()

include_directories(${MUJOCO_INCLUDE_DIR})

if(NOT MUJOCO_LIBRARY AND NOT TARGET mujoco::mujoco)
    message(FATAL_ERROR "Cannot find MuJoCo library in ${MUJOCO_LIB_DIR}")
endif()


set(blasfeo_INCLUDE_DIRS /home/ubuntu/.local/blasfeo/include)
set(hpipm_INCLUDE_DIRS /home/ubuntu/.local/hpipm/include)
set(blasfeo_LIBRARIES /home/ubuntu/.local/blasfeo/lib/libblasfeo.a)
set(hpipm_LIBRARIES /home/ubuntu/.local/hpipm/lib/libhpipm.a)

add_library(wm MODULE
    src/JointCommand.cpp
    src/JointState.cpp
    src/MPC.cpp
    src/RobotState.cpp
    src/SE3.cpp
    src/WalkingManager.cpp
    src/WholeBodyController.cpp
    src/utils.cpp 
    src/wm_binding.cpp
)
target_include_directories(wm PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    $ENV{HOME}/.local/include
    ${blasfeo_INCLUDE_DIRS}
    ${hpipm_INCLUDE_DIRS}
)

# WM
target_link_libraries(wm PRIVATE
    mujoco::mujoco
    pybind11::module
    pinocchio::pinocchio
    Eigen3::Eigen
    $ENV{HOME}/.local/lib/libhpipm.a
    $ENV{HOME}/.local/lib/libblasfeo.a
)

set_target_properties(wm PROPERTIES 
    PREFIX ""
)

install(TARGETS wm
        LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/compiled)
        
message(STATUS ">>> MODE: PYTHON, compilining wm.so)")

else()

message(STATUS ">>> MODE: C++ ")

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/exec)

find_package(glfw3 REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(mujoco CONFIG REQUIRED)

message(STATUS "MuJoCo found located at: ${mujoco_DIR}")

set(LOCAL_LIB_DIR "/home/ubuntu/.local/lib")
set(LOCAL_INC_DIR "/home/ubuntu/.local/include")

include_directories(BEFORE "${LOCAL_INC_DIR}")
include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Definiamo i file esatti che hai trovato con ls
set(blasfeo_LIBRARIES "${LOCAL_LIB_DIR}/libblasfeo.a")
set(hpipm_LIBRARIES   "${LOCAL_LIB_DIR}/libhpipm.a")

add_library(UtilsLib STATIC
    src/JointCommand.cpp
    src/JointState.cpp
    src/MPC.cpp
    src/RobotState.cpp
    src/SE3.cpp
    src/WalkingManager.cpp
    src/WholeBodyController.cpp
    src/utils.cpp
)

target_include_directories(UtilsLib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${hpipm_INCLUDE_DIRS}
    ${blasfeo_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
)

target_link_libraries(UtilsLib
    pinocchio::pinocchio
    mujoco::mujoco   # Link pulito a MuJoCo di Pixi
    ${hpipm_LIBRARIES}
    ${blasfeo_LIBRARIES}
)

# --------------------------------------------------------
# 4. DEFINIZIONE ESEGUIBILE (main)
# --------------------------------------------------------
add_executable(main main.cpp)

target_include_directories(main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(main
    glfw
    mujoco::mujoco
    UtilsLib
)

# --------------------------------------------------------
# 5. INSTALLAZIONE (Fondamentale per pixi build)
# --------------------------------------------------------
install(TARGETS main UtilsLib
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

# --------------------------------------------------------
# 6. SILENZIA WARNING
# --------------------------------------------------------
set(COAL_DISABLE_HPP_FCL_WARNINGS ON CACHE BOOL "" FORCE)
add_compile_definitions(COAL_DISABLE_HPP_FCL_WARNINGS)
target_compile_options(UtilsLib PRIVATE -Wno-deprecated-declarations -Wno-macro-redefined)
target_compile_options(main PRIVATE -Wno-deprecated-declarations -Wno-macro-redefined)

message(STATUS ">>> MODE: C++ ")

endif()
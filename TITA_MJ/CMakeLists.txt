cmake_minimum_required(VERSION 3.16)
project(TITA_MJ)

option(BUILD_PYTHON_BINDINGS "Decide if compile the bindings")
set(LOCAL_DEP_DIR "${PROJECT_SOURCE_DIR}/../dependencies/local")

if(BUILD_PYTHON_BINDINGS)
message(STATUS ">>> MODE: PYTHON, compilining wm.so)")

set(Python3_ROOT_DIR "/home/ubuntu/miniconda3/envs/tianshou_gpu")
find_package(Python COMPONENTS Interpreter Development REQUIRED)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "${PROJECT_SOURCE_DIR}/compiled;$ENV{CONDA_PREFIX}/lib;${PROJECT_SOURCE_DIR}/../dependencies/local/lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_PREFIX_PATH "/opt/openrobots" ${CMAKE_PREFIX_PATH})

# ---- Forza OpenRobots come priorità ----
list(PREPEND CMAKE_PREFIX_PATH "/opt/openrobots")
set(ENV{PKG_CONFIG_PATH} "/opt/openrobots/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")

# ---- RPATH: carica sempre da OpenRobots ----
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "/opt/openrobots/lib;${PROJECT_SOURCE_DIR}/compiled")

set(Python3_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python")
find_package(Python COMPONENTS Interpreter Development REQUIRED)

find_package(Boost REQUIRED COMPONENTS filesystem serialization system)
find_package(glfw3 REQUIRED)
find_package(pinocchio REQUIRED)
#find_package(crocoddyl REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(IPOPT ipopt)

if(IPOPT_FOUND)
    message(STATUS "Ipopt found: ${IPOPT_INCLUDE_DIRS}")
    include_directories(SYSTEM ${IPOPT_INCLUDE_DIRS})
    link_directories(${IPOPT_LIBRARY_DIRS})
else()
    # Se pkg-config fallisce, proviamo il path diretto di Pixi/Conda
    include_directories(SYSTEM "$ENV{CONDA_PREFIX}/include")
    include_directories(SYSTEM "$ENV{CONDA_PREFIX}/include/coin-or")
endif()

set(MUJOCO_PY_PATH "/home/ubuntu/.local/lib/python3.10/site-packages/mujoco")
list(APPEND CMAKE_INSTALL_RPATH ${MUJOCO_PY_PATH})

add_library(mujoco::mujoco UNKNOWN IMPORTED)
set_target_properties(mujoco::mujoco PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_PY_PATH}/libmujoco.so.3.3.7"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_PY_PATH}/include"
)

message(STATUS "Using MuJoCo from pip at ${MUJOCO_PY_PATH}")


set(blasfeo_INCLUDE_DIRS "${LOCAL_DEP_DIR}/include")
set(hpipm_INCLUDE_DIRS   "${LOCAL_DEP_DIR}/include")
set(blasfeo_LIBRARIES    "${LOCAL_DEP_DIR}/lib/libblasfeo.a")
set(hpipm_LIBRARIES      "${LOCAL_DEP_DIR}/lib/libhpipm.a")
set(CROCODYD_INCLUDE_DIR    "${LOCAL_DEP_DIR}/include/crocoddyl")
set(CROCODYD_LIB_DIR "${LOCAL_DEP_DIR}/lib")

set(CROCODYD_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/local/lib/libcrocoddyl.so")
set(HPIPM_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/local/lib/libhpipm.a")
set(BLASFEO_LIB "${CMAKE_CURRENT_SOURCE_DIR}/../dependencies/local/lib/libblasfeo.a")

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
    ${blasfeo_INCLUDE_DIRS}/include
    ${hpipm_INCLUDE_DIRS}
    ${blasfeo_INCLUDE_DIRS}
    ${hpipm_INCLUDE_DIRS}
    ${CROCODYD_INCLUDE_DIR}
)

# WM
target_link_libraries(wm PRIVATE
    ${Boost_LIBRARIES}
    ${CROCODYD_LIB}
    mujoco::mujoco
    pybind11::module
    pinocchio::pinocchio
    Eigen3::Eigen
    ${hpipm_LIBRARIES}
    ${blasfeo_LIBRARIES}
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

#set(LOCAL_DEP_DIR "/home/ubuntu/Desktop/repo_rl/Residual_Learning_TITA/dependencies/local")
set(LOCAL_DEP_DIR "${PROJECT_SOURCE_DIR}/../dependencies/local")

message(STATUS ">>> FORCING LOCAL DEPS FROM: ${LOCAL_DEP_DIR}")

# Inserisci il percorso all'inizio della ricerca di CMake
list(INSERT CMAKE_PREFIX_PATH 0 "${LOCAL_DEP_DIR}")

# Forza le directory di inclusione e i link per UtilsLib
include_directories(SYSTEM "${LOCAL_DEP_DIR}/include")
link_directories("${LOCAL_DEP_DIR}/lib")

find_package(glfw3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(crocoddyl REQUIRED)

set(MUJOCO_PY_PATH "/home/ubuntu/.local/lib/python3.10/site-packages/mujoco")
list(APPEND CMAKE_INSTALL_RPATH ${MUJOCO_PY_PATH})

add_library(mujoco::mujoco UNKNOWN IMPORTED)
set_target_properties(mujoco::mujoco PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_PY_PATH}/libmujoco.so.3.3.7"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_PY_PATH}/include"
)

message(STATUS "Using MuJoCo from pip at ${MUJOCO_PY_PATH}")

find_package(pinocchio REQUIRED)
message(STATUS ">>> Pinocchio found at: ${pinocchio_DIR}")
message(STATUS ">>> Pinocchio library: ${pinocchio_LIBRARIES}")
message(STATUS ">>> Pinocchio includes: ${pinocchio_INCLUDE_DIRS}")

if(NOT pinocchio_FOUND)
    message(STATUS "Pinocchio not found via find_package, using local lib")
    find_library(PINOCCHIO_LIB NAMES pinocchio PATHS "${LOCAL_DEP_DIR}/lib")
    if(NOT PINOCCHIO_LIB)
        message(FATAL_ERROR "Pinocchio C++ library not found in ${LOCAL_DEP_DIR}/lib")
    endif()
    add_library(pinocchio::pinocchio UNKNOWN IMPORTED)
    set_target_properties(pinocchio::pinocchio PROPERTIES IMPORTED_LOCATION ${PINOCCHIO_LIB})
    include_directories("${LOCAL_DEP_DIR}/include")
endif()

find_package(PkgConfig REQUIRED)
pkg_check_modules(IPOPT ipopt)

if(IPOPT_FOUND)
    message(STATUS "Ipopt found: ${IPOPT_INCLUDE_DIRS}")
    include_directories(SYSTEM ${IPOPT_INCLUDE_DIRS})
    link_directories(${IPOPT_LIBRARY_DIRS})
else()
    # Se pkg-config fallisce, proviamo il path diretto di Pixi/Conda
    include_directories(SYSTEM "$ENV{CONDA_PREFIX}/include")
    include_directories(SYSTEM "$ENV{CONDA_PREFIX}/include/coin-or")
endif()

message(STATUS "MuJoCo found located at: ${mujoco_DIR}")

set(LOCAL_LIB_DIR "${LOCAL_DEP_DIR}/lib")
set(LOCAL_INC_DIR "${LOCAL_DEP_DIR}/include")
set(CROCODYD_INCLUDE_DIR "${LOCAL_DEP_DIR}/include")
set(CROCODYD_LIB_DIR "${LOCAL_DEP_DIR}/lib")

include_directories(BEFORE "${LOCAL_INC_DIR}")
include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Definiamo i file esatti che hai trovato con ls
set(blasfeo_LIBRARIES "${LOCAL_LIB_DIR}/libblasfeo.a")
set(hpipm_LIBRARIES   "${LOCAL_LIB_DIR}/libhpipm.a")

add_library(UtilsLib SHARED
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
    ${CROCODYD_LIB_DIR}/libcrocoddyl.so
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
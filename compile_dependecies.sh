#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

GIT_ROOT=$(git rev-parse --show-toplevel)
cd ${GIT_ROOT}

echo "-------------------------------------------------------"
echo " Starting TITA Dependencies Setup (BLASFEO & HPIPM)    "
echo "-------------------------------------------------------"

echo ">>>     Creating dependencies directory..."
mkdir -p dependencies
cd dependencies
DEP_PATH=$(pwd)

echo ">>>     Cloning BLASFEO..."
if [ ! -d "blasfeo" ]; then
    git clone https://github.com/giaf/blasfeo
else
    echo "Blasfeo folder already exists, skipping clone."
fi

echo ">>>     Cloning HPIPM..."
if [ ! -d "hpipm" ]; then
    git clone https://github.com/giaf/hpipm
else
    echo "Hpipm folder already exists, skipping clone."
fi

# 3. Build BLASFEO
echo ">>>     Building BLASFEO..."
cd "$DEP_PATH/blasfeo"
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../../local -DTARGET=GENERIC -DCMAKE_C_FLAGS="-fPIC"
make -j$(nproc)
make install

echo ">>>     Organizing BLASFEO library files..."
cd "$DEP_PATH/blasfeo"
mkdir -p lib
cp build/libblasfeo.a lib/
echo "Success: libblasfeo.a moved to $(pwd)/lib/"

sleep 3

# 5. Build HPIPM
echo ">>>     Building HPIPM..."
cd "$DEP_PATH/hpipm"
mkdir -p build
cd build

INSTALL_DIR_ABS=$(readlink -f "../../local")
cmake .. -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR_ABS" \
         -DBLASFEO_PATH="$INSTALL_DIR_ABS" \
         -DTARGET=GENERIC
make -j$(nproc)
make install

echo ">>>     Organizing HPIPM library files..."
cd "$DEP_PATH/hpipm"
mkdir -p lib
cp build/libhpipm.a lib/

echo "-------------------------------------------------------"
echo " SETUP COMPLETED SUCCESSFULLY                         "
echo "-------------------------------------------------------"
echo "BLASFEO Path: $DEP_PATH/blasfeo/lib/libblasfeo.a"
echo "HPIPM Path:   $DEP_PATH/hpipm/lib/libhpipm.a"
echo "-------------------------------------------------------"


sleep 3

# 5. Build Crocoddyl
echo ">>>     Building Crocoddyl..."
cd "$DEP_PATH/crocoddyl"

# Directory di installazione assoluta (come per HPIPM)
INSTALL_DIR_ABS=$(readlink -f "../../local")

cmake .. \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR_ABS" \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
make install

echo "-------------------------------------------------------"
echo " Crocoddyl installed in: $INSTALL_DIR_ABS"
echo "-------------------------------------------------------"

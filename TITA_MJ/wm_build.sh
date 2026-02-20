#!/bin/bash
set -e  # esce se qualche comando fallisce

# Ottieni la cartella assoluta del progetto
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo ">>> Project dir: $PROJECT_DIR"

# Cancella il modulo Python compilato
if [ -f "$PROJECT_DIR/compiled/wm.so" ]; then
    echo ">>> Removing compiled/wm.so"
    rm "$PROJECT_DIR/compiled/wm.so"
fi

if [ -f "$PROJECT_DIR/build/wm.so" ]; then
    echo ">>> Removing build/wm.so"
    rm "$PROJECT_DIR/build/wm.so"
fi

# Vai nella cartella build e pulisci tutto
BUILD_DIR="$PROJECT_DIR/build"
echo ">>> Cleaning build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
rm -rf *

# CMake + make + make install
echo ">>> Running CMake..."
cmake .. -DBUILD_PYTHON_BINDINGS=ON -DPYBIND11_PYTHON_VERSION=3.10

echo ">>> Building..."
make -j$(nproc)

echo ">>> Installing to compiled/"
make install

# Torna alla cartella principale
cd "$PROJECT_DIR"
echo ">>> Done! wm.so is now in compiled/"

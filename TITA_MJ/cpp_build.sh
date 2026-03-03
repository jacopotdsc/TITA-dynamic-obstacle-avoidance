#!/bin/bash
set -e  # esce se qualche comando fallisce

GIT_ROOT="$(git rev-parse --show-toplevel)"
echo ">>> Git root: $GIT_ROOT"

# Vai nella cartella TITA_MJ
PROJECT_DIR="$GIT_ROOT/TITA_MJ"
echo ">>> Project dir: $PROJECT_DIR"
cd "$PROJECT_DIR"

# Cancella il modulo Python compilato
if [ -f "$PROJECT_DIR/exec/main" ]; then
    echo ">>> Removing exec/main"
    rm "$PROJECT_DIR/exec/main"
fi

# Vai nella cartella build e pulisci tutto
BUILD_DIR="$PROJECT_DIR/build"
echo ">>> Cleaning build directory: $BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
rm -rf *

# CMake + make + make install
echo ">>> Running CMake..."
cmake .. -DBUILD_PYTHON_BINDINGS=off

echo ">>> Building..."
make -j$(nproc)


# Torna alla cartella principale
cd "$PROJECT_DIR"
echo ">>> Done! main is now in exec/"
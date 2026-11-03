#!/bin/sh
# Compile net.c to ../net.wasm (clang 15 or later with the wasm32 target and wasm-ld; no libc needed).
set -e
cd "$(dirname "$0")"
clang --target=wasm32 -O3 -msimd128 -nostdlib -ffreestanding -fno-builtin-memset \
  -Wl,--no-entry -Wl,--export=__heap_base -Wl,--export-memory -Wl,--growable-table -Wl,-z,stack-size=65536 \
  -o ../net.wasm net.c
ls -l ../net.wasm

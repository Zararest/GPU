# Debug version
CLANG_PATH=~/projects/llvm-project/llvm/build/bin

clang -O3 -S -emit-llvm -o example.ll example.c
$CLANG_PATH/llc -pbqp-dump-graphs -pbqp-coalescing -regalloc=pbqp -O3 example.ll